from __future__ import annotations

import json
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService


class AnalysisGraphService:
    """Orquestrador híbrido profissional.

    Regra principal:
    - Nexus/RAG é a rota principal para perguntas abertas, contexto e explicações.
    - Tabular/CSV é ferramenta auxiliar para contagens, agrupamentos, listagens e filtros explícitos.
    - Se tabular não for aplicável ou falhar com baixa confiança, volta para RAG limpo.
    """

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
        # 1) Usa tabular apenas quando a pergunta realmente pedir consulta estruturada.
        tabular_result = self.tabular_service.answer_question(case_id, question, documents, mode=mode)
        if tabular_result:
            tabular_result.setdefault("route_priority", "tool_tabular")
            return tabular_result

        # 2) Caso contrário, RAG/Nexus limpo é a rota principal.
        return self._answer_with_rag(case_id, question, analysis, chat_history, mode)

    def _answer_with_rag(
        self,
        case_id: str,
        question: str,
        analysis: dict[str, Any],
        chat_history: list[dict[str, Any]] | None,
        mode: str,
    ) -> dict[str, Any]:
        evidences = self.retrieval_service.search(case_id, question, top_k=8)
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
                formatted["route_priority"] = "rag_primary"
                return formatted
        formatted["answer_text"] = formatted.get("summary", "")
        formatted["route"] = "document"
        formatted["query_type"] = "document_qa"
        formatted["route_priority"] = "rag_primary"
        return formatted

    def _ask_openai(
        self,
        question: str,
        analysis: dict[str, Any],
        evidences: list[dict[str, Any]],
        history: list[dict[str, str]],
        mode: str,
    ) -> str | None:
        evidence_blob = "\n\n".join([
            f"[{e.get('filename')} | score={e.get('score')}]\n{e.get('excerpt')}"
            for e in evidences
        ])[:16000]
        system_prompt = (
            "Você é um analista sênior de automação e arquitetura do GABBI. Responda em português do Brasil. "
            "Use apenas as evidências fornecidas e o contexto analítico do caso. "
            "Não use conhecimento externo para afirmar fatos do domínio do cliente. "
            "Quando as evidências forem insuficientes, diga exatamente o que faltou e sugira uma forma de consultar a base. "
            "Estruture a resposta em markdown, com títulos curtos, listas claras e conteúdo organizado. "
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

Evidências recuperadas do Nexus/RAG:
{evidence_blob}

Gere uma resposta organizada e baseada somente nas evidências recuperadas.
Se a pergunta pedir número total, agrupamento ou listagem estruturada e as evidências não trouxerem dados suficientes, informe que a consulta tabular deve ser usada.
"""
        return self.llm_service.generate_chat(system_prompt, history[-8:], user_prompt)
