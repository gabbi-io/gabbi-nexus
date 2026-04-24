from __future__ import annotations

import json
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService


class AnalysisGraphService:
    def __init__(self, retrieval_service, analysis_service):
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.llm_service = LLMService()
        self.tabular_service = TabularQueryService(llm_service=self.llm_service)

    def llm_status(self) -> dict[str, Any]:
        return self.llm_service.status()

    def build_tabular_catalog(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        return self.tabular_service.build_catalog(case_id, documents)

    def ask(
        self,
        case_id: str,
        question: str,
        analysis: dict[str, Any],
        documents: list[dict[str, Any]],
        chat_history: list[dict[str, Any]] | None = None,
        mode: str = "executive",
    ) -> dict[str, Any]:
        tabular_result = self.tabular_service.answer_question(case_id, question, documents, mode=mode)
        if tabular_result:
            return tabular_result

        evidences = self.retrieval_service.search(case_id, question, top_k=5)
        formatted = self.analysis_service.format_answer(question, evidences, analysis, mode=mode)
        history = []
        for item in chat_history or []:
            if item.get("question"):
                history.append({"role": "user", "content": item["question"]})
            if item.get("answer_text"):
                history.append({"role": "assistant", "content": item["answer_text"]})
        if self.llm_service.status()["enabled"]:
            answer = self._ask_openai(question, analysis, evidences, history, mode)
            if answer:
                formatted["summary"] = answer
                formatted["answer_text"] = answer
                formatted["route"] = "document"
                formatted["query_type"] = "document_qa"
                return formatted
        formatted["answer_text"] = formatted["summary"]
        formatted["route"] = "document"
        formatted["query_type"] = "document_qa"
        return formatted

    def _ask_openai(self, question: str, analysis: dict[str, Any], evidences: list[dict[str, Any]], history: list[dict[str, str]], mode: str) -> str | None:
        evidence_blob = "\n\n".join([f"[{e.get('filename')} | score={e.get('score')}]\n{e.get('excerpt')}" for e in evidences])[:12000]
        system_prompt = (
            "Você é um analista sênior de automação e arquitetura do GABBI. Responda em português do Brasil. "
            "Use apenas as evidências fornecidas e o contexto analítico do caso. "
            "Quando inferir algo, diga que se trata de inferência. Estruture a resposta em markdown, com títulos curtos, listas claras e conteúdo organizado. "
        )
        if mode == "executive":
            system_prompt += "Priorize linguagem executiva, objetiva e orientada à decisão."
        elif mode == "analytical":
            system_prompt += "Priorize análise detalhada, riscos, regras, exceções e automação recomendada."
        else:
            system_prompt += "Priorize precisão técnica e inclua observações sobre evidências utilizadas."
        user_prompt = f"""
Pergunta do usuário:
{question}

Contexto analítico já calculado:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

Evidências recuperadas:
{evidence_blob}

Gere uma resposta organizada com, quando aplicável:
- Resumo executivo
- Objetivo do documento/processo
- Processos de negócio identificados
- Regras explícitas e implícitas
- Riscos e gargalos
- Melhor automação inicial no GABBI
- Próximo passo recomendado

Não invente detalhes fora das evidências.
"""
        return self.llm_service.generate_chat(system_prompt, history, user_prompt)
