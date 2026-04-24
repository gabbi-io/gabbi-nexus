from __future__ import annotations

import json
import re
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
        specialist_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = specialist_state or self._initial_specialist_state()
        discovery = self._handle_discovery(question, analysis, documents, chat_history or [], state)
        if discovery["status"] != "confirmed":
            return {
                "route": "discovery",
                "query_type": "context_alignment",
                "answer_text": discovery["answer_text"],
                "summary": discovery["answer_text"],
                "evidences": [],
                "evidence_files": [],
                "specialist_state": discovery["state"],
                "technical": {"discovery": discovery},
            }

        state = discovery["state"]
        active_question = discovery.get("active_question") or question
        specialist_role = state.get("specialist_role") or "especialista do domínio"

        tabular_result = self.tabular_service.answer_question(
            case_id, active_question, documents, mode=mode, specialist_role=specialist_role
        )
        if tabular_result:
            tabular_result["specialist_state"] = state
            tabular_result["answer_text"] = self._prepend_specialist_context(tabular_result["answer_text"], state)
            tabular_result["summary"] = tabular_result["answer_text"]
            return tabular_result

        evidences = self.retrieval_service.search(case_id, active_question, top_k=5)
        formatted = self.analysis_service.format_answer(active_question, evidences, analysis, mode=mode)
        history = []
        for item in chat_history or []:
            if item.get("question"):
                history.append({"role": "user", "content": item["question"]})
            if item.get("answer_text"):
                history.append({"role": "assistant", "content": item["answer_text"]})
        if self.llm_service.status()["enabled"]:
            answer = self._ask_openai(active_question, analysis, evidences, history, mode, state)
            if answer:
                formatted["summary"] = answer
                formatted["answer_text"] = answer
                formatted["route"] = "document"
                formatted["query_type"] = "document_qa"
                formatted["specialist_state"] = state
                return formatted
        formatted["answer_text"] = self._prepend_specialist_context(formatted["summary"], state)
        formatted["summary"] = formatted["answer_text"]
        formatted["route"] = "document"
        formatted["query_type"] = "document_qa"
        formatted["specialist_state"] = state
        return formatted

    def _initial_specialist_state(self) -> dict[str, Any]:
        return {
            "status": "discovering",
            "domain": None,
            "user_goal": None,
            "decision_need": None,
            "output_preference": None,
            "specialist_role": None,
            "understanding_summary": None,
            "pending_request": None,
            "missing_fields": ["user_goal", "decision_need", "output_preference"],
            "confidence": 0.0,
        }

    def _handle_discovery(
        self,
        question: str,
        analysis: dict[str, Any],
        documents: list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        qnorm = self._norm(question)
        if not state.get("pending_request") and self._is_substantive_request(qnorm):
            state["pending_request"] = question

        if state.get("status") == "confirmed" and self._looks_like_revision(qnorm):
            state["status"] = "discovering"

        if state.get("status") != "confirmed" and self._looks_like_confirmation(qnorm):
            state["status"] = "confirmed"
            state["confidence"] = max(state.get("confidence", 0.0), 0.9)
            return {"status": "confirmed", "state": state, "active_question": state.get("pending_request") or question}

        update = self._infer_specialist_state(question, analysis, documents, chat_history, state)
        state.update({k: v for k, v in update.items() if v not in (None, [], "")})
        missing = [f for f in ["domain", "user_goal", "decision_need", "output_preference", "specialist_role"] if not state.get(f)]
        state["missing_fields"] = missing

        if self._looks_like_revision(qnorm):
            state["status"] = "discovering"
            return {"status": state["status"], "state": state, "answer_text": self._build_clarifying_message(state, revised=True)}

        if missing:
            state["status"] = "discovering"
            return {"status": state["status"], "state": state, "answer_text": self._build_clarifying_message(state)}

        state["understanding_summary"] = self._build_understanding_summary(state)
        state["status"] = "awaiting_confirmation"
        return {"status": state["status"], "state": state, "answer_text": self._build_confirmation_prompt(state)}

    def _infer_specialist_state(
        self,
        question: str,
        analysis: dict[str, Any],
        documents: list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        if self.llm_service.status().get("enabled"):
            prompt = self._build_discovery_prompt(question, analysis, documents, chat_history, state)
            payload = self.llm_service.generate_json(
                "Você é um orquestrador de descoberta do GABBI. Retorne JSON com domain, user_goal, decision_need, output_preference, specialist_role e confidence(0-1). Seja conservador e use null quando não souber.",
                prompt,
            )
            if payload:
                return payload
        return self._infer_with_heuristics(question, analysis, documents, state)

    def _build_discovery_prompt(
        self,
        question: str,
        analysis: dict[str, Any],
        documents: list[dict[str, Any]],
        chat_history: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> str:
        docs = [{"filename": d.get("filename"), "content_type": d.get("content_type")} for d in documents[:10]]
        recent = chat_history[-6:]
        return f"""
Mensagem mais recente do usuário: {question}

Estado atual: {json.dumps(state, ensure_ascii=False)}

Análise do caso: {json.dumps(analysis, ensure_ascii=False)}

Documentos do caso: {json.dumps(docs, ensure_ascii=False)}

Histórico recente: {json.dumps(recent, ensure_ascii=False)}

Objetivo: identificar o domínio, a necessidade do usuário, o tipo de decisão que ele quer tomar, a preferência de saída e qual especialista virtual deve conduzir a resposta final.
"""

    def _infer_with_heuristics(
        self,
        question: str,
        analysis: dict[str, Any],
        documents: list[dict[str, Any]],
        state: dict[str, Any],
    ) -> dict[str, Any]:
        q = self._norm(question)
        domain = state.get("domain")
        specialist = state.get("specialist_role")
        if any(x in q for x in ["incidente", "sla", "ticket", "prioridade", "canal", "mudanca"]):
            domain = domain or "Operações e gestão de incidentes"
            specialist = specialist or "especialista em operações, incidentes e continuidade"
        elif any(x in q for x in ["contrato", "clausula", "multa", "vigencia", "jurid"]):
            domain = domain or "Jurídico e contratos"
            specialist = specialist or "especialista em contratos, cláusulas e risco jurídico"
        elif any(x in q for x in ["financeiro", "receita", "despesa", "custo", "margem"]):
            domain = domain or "Finanças e performance"
            specialist = specialist or "especialista em finanças, performance e indicadores"
        elif any(x in q for x in ["rh", "colaborador", "funcionario", "desligamento", "folha"]):
            domain = domain or "RH e gestão de pessoas"
            specialist = specialist or "especialista em RH, pessoas e indicadores de gestão"
        elif any(x in q for x in ["saude", "beneficiario", "guia", "senha", "prestador"]):
            domain = domain or "Saúde suplementar e operação assistencial"
            specialist = specialist or "especialista em saúde suplementar, operação e processo assistencial"
        else:
            doc_names = " ".join((d.get("filename") or "") for d in documents).lower()
            if "incidente" in doc_names:
                domain = domain or "Operações e gestão de incidentes"
                specialist = specialist or "especialista em operações e incidentes"

        goal = state.get("user_goal")
        if not goal:
            if any(x in q for x in ["automatizar", "automacao", "fluxo", "workflow"]):
                goal = "identificar o que automatizar e como estruturar a solução"
            elif any(x in q for x in ["resuma", "resumo", "teor", "entender", "explica"]):
                goal = "entender o teor do material e seus principais pontos"
            elif any(x in q for x in ["quantos", "quais", "listar", "contar"]):
                goal = "consultar dados específicos e obter respostas precisas"
            else:
                goal = "entender o material para apoiar uma decisão"

        decision = state.get("decision_need")
        if not decision:
            if any(x in q for x in ["decidir", "priorizar", "vale a pena", "recomenda"]):
                decision = "priorização e tomada de decisão"
            else:
                decision = "entendimento do contexto antes de definir a próxima ação"

        output = state.get("output_preference")
        if not output:
            if any(x in q for x in ["executivo", "diretoria", "gestor"]):
                output = "visão executiva"
            elif any(x in q for x in ["tecnico", "arquitetura", "detalhe"]):
                output = "visão técnica"
            else:
                output = "visão orientada a decisão com linguagem clara"

        return {
            "domain": domain,
            "specialist_role": specialist,
            "user_goal": goal,
            "decision_need": decision,
            "output_preference": output,
            "confidence": 0.72 if domain and specialist else 0.55,
        }

    def _build_clarifying_message(self, state: dict[str, Any], revised: bool = False) -> str:
        missing = state.get("missing_fields") or []
        intro = "## Vamos alinhar o contexto antes de responder como especialista\n\n"
        if revised:
            intro = "## Entendi, vamos ajustar o entendimento\n\nObrigado pela correção. Vou recalibrar o contexto antes de seguir.\n\n"
        prompts = []
        if "user_goal" in missing:
            prompts.append("- **Qual é o seu objetivo principal com esse material?** Ex.: resumir, decidir, comparar, automatizar, entender riscos ou extrair dados.")
        if "decision_need" in missing:
            prompts.append("- **Que tipo de decisão ou encaminhamento você quer apoiar?** Ex.: priorização, risco, operação, custo, conformidade, melhoria de processo.")
        if "output_preference" in missing:
            prompts.append("- **Você prefere uma visão executiva, técnica ou orientada à ação?**")
        if "domain" in missing or "specialist_role" in missing:
            prompts.append("- **Se houver uma área mais adequada para conduzir a análise, qual seria?** Ex.: operações, jurídico, finanças, RH, saúde, compras. Se quiser, eu posso inferir isso para você.")
        current = []
        if state.get("domain"):
            current.append(f"- Domínio percebido até agora: **{state['domain']}**")
        if state.get("specialist_role"):
            current.append(f"- Especialista mais provável: **{state['specialist_role']}**")
        current_block = "\n".join(current)
        head = f"### O que já entendi\n{current_block}\n\n" if current_block else ""
        return intro + head + "### Para eu fechar o entendimento\n" + "\n".join(prompts)

    def _build_understanding_summary(self, state: dict[str, Any]) -> str:
        return (
            f"Vou atuar como **{state.get('specialist_role', 'especialista do domínio')}**. "
            f"Entendi que o contexto é **{state.get('domain', 'domínio não definido')}**, "
            f"que seu objetivo é **{state.get('user_goal', 'não definido')}**, "
            f"e que a resposta deve apoiar **{state.get('decision_need', 'não definido')}**, "
            f"com saída em **{state.get('output_preference', 'formato não definido')}**."
        )

    def _build_confirmation_prompt(self, state: dict[str, Any]) -> str:
        return (
            "## Validação do entendimento\n\n"
            f"{state.get('understanding_summary')}\n\n"
            "Se estiver correto, responda algo como **'sim, pode seguir'**. "
            "Se quiser ajustar, responda em linguagem natural, por exemplo: **'não é bem jurídico, é mais operação e risco'**."
        )

    def _ask_openai(
        self,
        question: str,
        analysis: dict[str, Any],
        evidences: list[dict[str, Any]],
        history: list[dict[str, str]],
        mode: str,
        state: dict[str, Any],
    ) -> str | None:
        evidence_blob = "\n\n".join([f"[{e.get('filename')} | score={e.get('score')}]\n{e.get('excerpt')}" for e in evidences])[:12000]
        specialist_role = state.get("specialist_role") or "especialista do domínio"
        system_prompt = (
            f"Você é um {specialist_role} dentro do GABBI. Responda em português do Brasil. "
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

Contexto confirmado do usuário:
- Domínio: {state.get('domain')}
- Objetivo: {state.get('user_goal')}
- Decisão a apoiar: {state.get('decision_need')}
- Formato desejado: {state.get('output_preference')}

Contexto analítico já calculado:
{json.dumps(analysis, ensure_ascii=False, indent=2)}

Evidências recuperadas:
{evidence_blob}

Gere uma resposta organizada e aderente ao especialista definido.
Não invente detalhes fora das evidências.
"""
        answer = self.llm_service.generate_chat(system_prompt, history, user_prompt)
        if answer:
            return self._prepend_specialist_context(answer, state)
        return None

    def _prepend_specialist_context(self, answer: str, state: dict[str, Any]) -> str:
        specialist = state.get("specialist_role")
        if not specialist:
            return answer
        prefix = (
            f"> **Especialista ativo:** {specialist}\n"
            f"> **Objetivo alinhado:** {state.get('user_goal', '-')}\n\n"
        )
        return prefix + answer

    def _looks_like_confirmation(self, q: str) -> bool:
        return any(x in q for x in ["sim pode seguir", "pode seguir", "isso mesmo", "correto", "confirmo", "pode responder", "esta correto", "ta correto"])

    def _looks_like_revision(self, q: str) -> bool:
        return any(x in q for x in ["nao e bem isso", "nao exatamente", "corrigindo", "na verdade", "ajustando", "nao seria"]) 

    def _is_substantive_request(self, q: str) -> bool:
        return not self._looks_like_confirmation(q) and len(q.split()) >= 4

    def _norm(self, value: str) -> str:
        q = value.lower().strip()
        repl = {"ç": "c", "ã": "a", "á": "a", "à": "a", "â": "a", "é": "e", "ê": "e", "í": "i", "ó": "o", "ô": "o", "õ": "o", "ú": "u"}
        for a, b in repl.items():
            q = q.replace(a, b)
        return re.sub(r"\s+", " ", q)
