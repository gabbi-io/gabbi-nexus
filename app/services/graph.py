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
        # Contexto conversacional deve ser isolado por conversa, não apenas por case/agent.
        # A base indexada e os CSVs permanecem compartilhados no case; somente filtros/follow-up
        # tabulares são resetados quando a conversa começa limpa.
        history = []
        for item in chat_history or []:
            if item.get("question"):
                history.append({"role": "user", "content": item["question"]})
            if item.get("answer_text"):
                history.append({"role": "assistant", "content": item["answer_text"]})

        conversation_key = self._conversation_context_key(case_id, chat_history or [])
        reset_context = len(chat_history or []) == 0

        # 1) Inteligência tabular/analítica primeiro, sem conhecimento externo.
        tabular_result = self.tabular_service.answer_question(
            case_id,
            question,
            documents,
            mode=mode,
            context_key=conversation_key,
            reset_context=reset_context,
        )
        if tabular_result and not tabular_result.get("should_fallback_to_rag"):
            return tabular_result

        # 2) RAG/document QA limpo quando:
        # - a pergunta não for tabular; ou
        # - a tentativa tabular zerar por contexto/follow-up potencialmente contaminado.
        evidences = self.retrieval_service.search(case_id, question, top_k=12)
        formatted = self.analysis_service.format_answer(question, evidences, analysis, mode=mode)
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

    def _conversation_context_key(self, case_id: str, chat_history: list[dict[str, Any]]) -> str:
        """Gera chave de memória tabular por conversa.

        Se o front iniciar uma nova conversa e enviar histórico vazio, a chave volta limpa.
        Quando houver histórico, usamos as primeiras mensagens para diferenciar conversas mesmo
        quando o mesmo case_id/agent é reaproveitado para a base.
        """
        if not chat_history:
            return f"{case_id}:fresh"
        seeds = []
        for item in chat_history[:4]:
            q = item.get("question") or ""
            a = item.get("answer_text") or ""
            seeds.append((q + "|" + a)[:250])
        raw = "\n".join(seeds)
        try:
            import hashlib
            digest = hashlib.sha1(raw.encode("utf-8", errors="ignore")).hexdigest()[:12]
        except Exception:
            digest = str(abs(hash(raw)))[:12]
        return f"{case_id}:{digest}"

    def _ask_openai(self, question: str, analysis: dict[str, Any], evidences: list[dict[str, Any]], history: list[dict[str, str]], mode: str) -> str | None:
        evidence_blob = "\n\n".join([f"[{e.get('filename')} | score={e.get('score')}]\n{e.get('excerpt')}" for e in evidences])[:16000]
        system_prompt = (
            "Você é um analista sênior de automação e arquitetura do GABBI. Responda em português do Brasil. "
            "Use apenas as evidências fornecidas e o contexto analítico do caso. "
            "Não use conhecimento externo quando a resposta depender da base. "
            "Se as evidências forem insuficientes, explique a limitação com precisão, mas não invente dados. "
            "Quando inferir algo, diga que se trata de inferência. Estruture a resposta em markdown. "
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
