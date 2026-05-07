from __future__ import annotations

import json
import os
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService


class AnalysisGraphService:
    """
    Orquestrador principal do chat do Nexus.

    Ajuste aplicado:
    - remove dependência de CSV/cache como fonte principal;
    - permite consulta tabular apenas via fonte viva configurada, preferencialmente PostgreSQL;
    - mantém histórico conversacional para fluidez;
    - mantém RAG/LLM como síntese final quando necessário.
    """

    def __init__(self, retrieval_service, analysis_service):
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.llm_service = LLMService()
        self.tabular_service = TabularQueryService(llm_service=self.llm_service)
        self.force_rag_first = os.getenv("GABBI_FORCE_RAG_FIRST", "false").lower() in {"1", "true", "yes", "sim"}

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
        history = self._build_history(chat_history)

        # RAG primeiro é opcional. Para perguntas quantitativas, normalmente o tabular DB é mais preciso.
        evidences: list[dict[str, Any]] = []
        if self.force_rag_first:
            evidences = self.retrieval_service.search(case_id, question, top_k=8)

        tabular_result = self.tabular_service.answer_question(
            case_id=case_id,
            question=question,
            documents=documents,
            mode=mode,
        )

        # Se a consulta tabular foi executada em fonte viva, pode responder diretamente.
        # O serviço tabular ajustado não usa mais gabbi_knowledge_table_active.csv como fonte.
        if tabular_result and not tabular_result.get("fallback_to_rag"):
            return tabular_result

        if not evidences:
            evidences = self.retrieval_service.search(case_id, question, top_k=8)

        formatted = self.analysis_service.format_answer(question, evidences, analysis, mode=mode)

        if self.llm_service.status()["enabled"]:
            answer = self._ask_openai(
                question=question,
                analysis=analysis,
                evidences=evidences,
                history=history,
                mode=mode,
                tabular_attempt=tabular_result,
            )
            if answer:
                formatted["summary"] = answer
                formatted["answer_text"] = answer
                formatted["route"] = "document"
                formatted["query_type"] = "document_qa"
                if tabular_result:
                    formatted["tabular_attempt"] = tabular_result.get("technical")
                return formatted

        formatted["answer_text"] = formatted.get("summary", "")
        formatted["route"] = "document"
        formatted["query_type"] = "document_qa"
        if tabular_result:
            formatted["tabular_attempt"] = tabular_result.get("technical")
        return formatted

    def _build_history(self, chat_history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for item in chat_history or []:
            if item.get("question"):
                history.append({"role": "user", "content": str(item["question"])})
            if item.get("answer_text"):
                history.append({"role": "assistant", "content": str(item["answer_text"])})
        return history[-8:]

    def _ask_openai(
        self,
        question: str,
        analysis: dict[str, Any],
        evidences: list[dict[str, Any]],
        history: list[dict[str, str]],
        mode: str,
        tabular_attempt: dict[str, Any] | None = None,
    ) -> str | None:
        evidence_blob = "\n\n".join(
            [
                f"[{e.get('filename')} | score={e.get('score')}]\n{e.get('excerpt')}"
                for e in evidences
            ]
        )[:16000]

        tabular_blob = ""
        if tabular_attempt:
            tabular_blob = "\n\nTentativa tabular anterior, apenas para diagnóstico. Não use filtros antigos como contexto permanente:\n"
            tabular_blob += json.dumps(
                tabular_attempt.get("technical", tabular_attempt),
                ensure_ascii=False,
                indent=2,
                default=str,
            )[:6000]

        system_prompt = (
            "Você é um analista sênior do GABBI. Responda em português do Brasil. "
            "Use apenas as evidências fornecidas, o contexto analítico do caso e o histórico recente. "
            "Se não houver evidência suficiente, diga isso de forma objetiva. Não invente dados. "
            "Quando fizer inferência, sinalize como inferência. Estruture em markdown claro. "
            "Nunca trate CSV/cache local como fonte da verdade."
        )
        if mode == "executive":
            system_prompt += " Priorize linguagem executiva, objetiva e orientada à decisão."
        elif mode == "analytical":
            system_prompt += " Priorize análise detalhada, riscos, regras, exceções e automação recomendada."
        else:
            system_prompt += " Priorize precisão técnica."

        user_prompt = f"""
Pergunta do usuário:
{question}

Contexto analítico do caso:
{json.dumps(analysis, ensure_ascii=False, indent=2, default=str)[:8000]}

Evidências recuperadas pelo RAG/Nexus:
{evidence_blob}
{tabular_blob}

Gere uma resposta organizada. Não use filtros tabulares antigos como se fossem contexto permanente.
"""
        return self.llm_service.generate_chat(system_prompt, history[-8:], user_prompt)
