from __future__ import annotations

import json
import os
import re
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService
from app.services.knowledge_structured_store import KnowledgeStructuredStore


class AnalysisGraphService:
    """
    Orquestrador principal do chat do Nexus.

    Ajuste aplicado:
    - considera sempre o contexto do case/upload atual;
    - considera também base treinada/PostgreSQL quando disponível;
    - não deixa a base Postgres substituir automaticamente o arquivo anexado;
    - remove CSV/cache local como fonte da verdade;
    - mantém histórico conversacional para fluidez.
    """

    FILE_FOCUS_TERMS = {
        "arquivo", "arquivos", "anexo", "anexado", "anexados", "documento", "documentos",
        "planilha", "excel", "pdf", "docx", "csv", "upload", "uploads", "enviado", "enviados",
        "neste", "nessa", "nesta", "desse", "deste", "dessa", "desta", "material"
    }

    KNOWLEDGE_FOCUS_TERMS = {
        "base", "conhecimento", "treinamento", "treinamentos", "treinado", "treinada",
        "article", "artigo", "artigos", "topic", "topico", "tópico", "historico", "histórico",
        "memoria", "memória", "gabbi", "nexus"
    }

    def __init__(self, retrieval_service, analysis_service):
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.llm_service = LLMService()
        self.tabular_service = TabularQueryService(llm_service=self.llm_service)
        self.knowledge_structured_store = KnowledgeStructuredStore()
        self.force_rag_first = self._env_bool("GABBI_FORCE_RAG_FIRST", True)
        self.always_synthesize_hybrid = self._env_bool("GABBI_ALWAYS_SYNTHESIZE_HYBRID", True)

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "sim", "on"}

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
        focus = self._detect_focus(question, documents)

        # 0. Engine analítica estruturada do Nexus (DuckDB).
        # Para count/list/group/distinct da base sincronizada, não usa RAG, memória ou amostra parcial.
        try:
            structured_result = self.knowledge_structured_store.answer_question(
                case_id=case_id,
                question=question,
                chat_history=chat_history or [],
            )
            if structured_result and not structured_result.get("fallback_to_rag"):
                return structured_result
        except Exception:
            pass

        # 1. Recupera evidências do case atual SEMPRE que houver documentos.
        # Isso evita o comportamento de responder apenas pela base treinada quando o usuário anexou arquivo.
        case_evidences: list[dict[str, Any]] = []
        if documents:
            case_evidences = self.retrieval_service.search(case_id, question, top_k=10)
            case_evidences = self._tag_evidences(case_evidences, default_source="case_upload")

        # 2. Executa consulta tabular híbrida.
        # O serviço tabular decide entre arquivo do case e Postgres, sem usar CSV cache como fonte da verdade.
        tabular_result = self.tabular_service.answer_question(
            case_id=case_id,
            question=question,
            documents=documents,
            mode=mode,
            source_preference=focus,
        )

        tabular_evidences = self._extract_tabular_evidences(tabular_result)

        # 3. Mescla evidências: upload primeiro quando o foco for arquivo/case; base primeiro quando o foco for treinamento/base.
        evidences = self._merge_evidences(
            case_evidences=case_evidences,
            tabular_evidences=tabular_evidences,
            focus=focus,
        )

        # 4. Só responde diretamente via tabular quando NÃO houver evidência de arquivo relevante
        # ou quando a própria fonte tabular for arquivo do case.
        if (
            tabular_result
            and not tabular_result.get("fallback_to_rag")
            and not self.always_synthesize_hybrid
            and self._can_return_tabular_direct(tabular_result, case_evidences, focus)
        ):
            return tabular_result

        formatted = self.analysis_service.format_answer(question, evidences, analysis, mode=mode)

        if self.llm_service.status()["enabled"]:
            answer = self._ask_openai(
                question=question,
                analysis=analysis,
                evidences=evidences,
                history=history,
                mode=mode,
                tabular_attempt=tabular_result,
                focus=focus,
            )
            if answer:
                formatted["summary"] = answer
                formatted["answer_text"] = answer
                formatted["route"] = "hybrid"
                formatted["query_type"] = "hybrid_qa"
                formatted["sources"] = {
                    "focus": focus,
                    "case_uploads": bool(case_evidences),
                    "tabular": bool(tabular_result and not tabular_result.get("fallback_to_rag")),
                    "tabular_source_type": self._tabular_source_type(tabular_result),
                    "conversation_memory": bool(history),
                }
                if tabular_result:
                    formatted["tabular_attempt"] = tabular_result.get("technical")
                return formatted

        # Fallback sem LLM.
        if tabular_result and not tabular_result.get("fallback_to_rag"):
            tabular_result["route"] = "hybrid_tabular"
            tabular_result["sources"] = {
                "focus": focus,
                "case_uploads": bool(case_evidences),
                "tabular_source_type": self._tabular_source_type(tabular_result),
            }
            return tabular_result

        formatted["answer_text"] = formatted.get("summary", "")
        formatted["route"] = "hybrid"
        formatted["query_type"] = "hybrid_qa"
        formatted["sources"] = {
            "focus": focus,
            "case_uploads": bool(case_evidences),
            "tabular": bool(tabular_result),
            "conversation_memory": bool(history),
        }
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

    def _detect_focus(self, question: str, documents: list[dict[str, Any]]) -> str:
        q = self._norm(question)
        tokens = set(q.split())
        has_docs = bool(documents)
        file_focus = bool(tokens & self.FILE_FOCUS_TERMS)
        kb_focus = bool(tokens & self.KNOWLEDGE_FOCUS_TERMS)

        if file_focus and has_docs:
            return "case_upload_first"
        if kb_focus and not file_focus:
            return "knowledge_base_first"
        if has_docs:
            return "hybrid_case_first"
        return "knowledge_base_first"

    def _merge_evidences(
        self,
        case_evidences: list[dict[str, Any]],
        tabular_evidences: list[dict[str, Any]],
        focus: str,
    ) -> list[dict[str, Any]]:
        case = self._tag_evidences(case_evidences, "case_upload")
        tabular = tabular_evidences or []

        if focus in {"case_upload_first", "hybrid_case_first"}:
            merged = case + tabular
        else:
            merged = tabular + case

        # Dedup simples por filename/chunk/type.
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for e in merged:
            key = (str(e.get("filename", "")), str(e.get("chunk_id", "")), str(e.get("type", "")))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out[:16]

    def _tag_evidences(self, evidences: list[dict[str, Any]], default_source: str) -> list[dict[str, Any]]:
        out = []
        for item in evidences or []:
            cloned = dict(item)
            metadata = dict(cloned.get("metadata") or {})
            metadata.setdefault("source", default_source)
            cloned["metadata"] = metadata
            out.append(cloned)
        return out

    def _extract_tabular_evidences(self, tabular_result: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not tabular_result:
            return []

        technical = tabular_result.get("technical") or {}
        execution = technical.get("execution") or {}
        if not execution:
            return []

        table = execution.get("table") or {}
        source_type = table.get("source_type") or table.get("source") or "tabular"
        filename = table.get("source") or table.get("filename") or "tabular"

        excerpt_payload = {
            "type": execution.get("type"),
            "count": execution.get("count"),
            "rows_considered": execution.get("rows_considered"),
            "rows_filtered": execution.get("rows_filtered"),
            "filters": execution.get("filters"),
            "preview": execution.get("preview") or execution.get("results"),
            "table": table,
        }

        return [
            {
                "filename": filename,
                "chunk_id": "tabular_execution",
                "type": "tabular",
                "score": 1.0,
                "excerpt": json.dumps(excerpt_payload, ensure_ascii=False, default=str)[:5000],
                "metadata": {
                    "source": "tabular",
                    "source_type": source_type,
                },
            }
        ]

    def _can_return_tabular_direct(self, tabular_result: dict[str, Any], case_evidences: list[dict[str, Any]], focus: str) -> bool:
        source_type = self._tabular_source_type(tabular_result)
        if source_type == "file":
            return True
        if source_type == "postgres" and not case_evidences:
            return True
        if focus == "knowledge_base_first":
            return True
        return False

    def _tabular_source_type(self, tabular_result: dict[str, Any] | None) -> str | None:
        if not tabular_result:
            return None
        execution = ((tabular_result.get("technical") or {}).get("execution") or {})
        table = execution.get("table") or {}
        return table.get("source_type")

    def _ask_openai(
        self,
        question: str,
        analysis: dict[str, Any],
        evidences: list[dict[str, Any]],
        history: list[dict[str, str]],
        mode: str,
        tabular_attempt: dict[str, Any] | None = None,
        focus: str = "hybrid_case_first",
    ) -> str | None:
        evidence_blob = "\n\n".join(
            [
                f"[{e.get('filename')} | tipo={e.get('type')} | score={e.get('score')} | fonte={(e.get('metadata') or {}).get('source') or (e.get('metadata') or {}).get('source_type')} ]\n{e.get('excerpt')}"
                for e in evidences
            ]
        )[:18000]

        tabular_blob = ""
        if tabular_attempt:
            tabular_blob = "\n\nResultado/diagnóstico tabular, quando houver. Use apenas se estiver coerente com as evidências do upload e da base:\n"
            tabular_blob += json.dumps(
                tabular_attempt.get("technical", tabular_attempt),
                ensure_ascii=False,
                indent=2,
                default=str,
            )[:7000]

        system_prompt = """
Você é um analista sênior do GABBI. Responda em português do Brasil.

REGRAS OBRIGATÓRIAS:
1. Nunca ignore arquivos anexados no case atual.
2. Quando houver arquivo/documento/planilha/PDF enviado no front, trate esse conteúdo como contexto prioritário do case.
3. A base treinada/PostgreSQL/Article deve complementar o raciocínio, não substituir o arquivo anexado.
4. Se a pergunta falar explicitamente de base, conhecimento, treinamento, artigos ou histórico, priorize a base treinada, mas ainda considere anexos relevantes.
5. Nunca trate CSV/cache local gabbi_knowledge_table_active_*.csv como fonte da verdade.
6. Use apenas evidências fornecidas e o histórico recente.
7. Não invente dados, números, filtros ou registros.
8. Se houver divergência entre arquivo anexado e base histórica, sinalize a divergência e priorize o arquivo atual para perguntas sobre o material enviado.
9. Explique de forma objetiva quais fontes foram consideradas: arquivo do case, base treinada/PostgreSQL e histórico conversacional.
10. Se não houver evidência suficiente, diga isso claramente.
""".strip()

        if mode == "executive":
            system_prompt += "\nPriorize linguagem executiva, objetiva e orientada à decisão."
        elif mode == "analytical":
            system_prompt += "\nPriorize análise detalhada, riscos, regras, exceções e automação recomendada."
        else:
            system_prompt += "\nPriorize precisão técnica."

        user_prompt = f"""
Foco de roteamento detectado: {focus}

Pergunta do usuário:
{question}

Contexto analítico do caso:
{json.dumps(analysis, ensure_ascii=False, indent=2, default=str)[:8000]}

Evidências híbridas recuperadas pelo Nexus:
{evidence_blob}
{tabular_blob}

Gere uma resposta organizada em markdown.
"""
        return self.llm_service.generate_chat(system_prompt, history[-8:], user_prompt)

    @staticmethod
    def _norm(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9_:/\-\s\.áàâãéêíóôõúç]+", " ", text)
        return " ".join(text.split())
