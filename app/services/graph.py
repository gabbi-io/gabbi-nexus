from __future__ import annotations

import json
import os
import re
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService


class AnalysisGraphService:
    """
    Orquestrador principal do chat do Nexus.

    Ajustes principais:
    - sempre considera os documentos do case atual/upload;
    - reindexa o case no momento do ask quando o índice em memória não existir;
    - evita resposta tabular direta quando a pergunta é sobre arquivo/anexo;
    - mantém histórico conversacional;
    - usa base treinada/tabular como complemento, não como substituta do arquivo anexado.
    """

    FILE_FOCUS_TERMS = {
        "arquivo", "arquivos", "anexo", "anexos", "anexado", "anexados", "documento", "documentos",
        "planilha", "excel", "pdf", "docx", "csv", "txt", "upload", "uploads", "enviado", "enviados",
        "neste", "nessa", "nesta", "desse", "deste", "dessa", "desta", "material", "conteudo", "conteúdo",
        "escopo do arquivo", "sobre o arquivo",
    }

    KNOWLEDGE_FOCUS_TERMS = {
        "base", "conhecimento", "treinamento", "treinamentos", "treinado", "treinada",
        "article", "artigo", "artigos", "topic", "topico", "tópico", "historico", "histórico",
        "memoria", "memória", "gabbi", "nexus", "knowledge",
    }

    def __init__(self, retrieval_service, analysis_service):
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.llm_service = LLMService()
        self.tabular_service = TabularQueryService(llm_service=self.llm_service)
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

        # Garante que o índice do case exista no processo atual.
        # Isso resolve cenários com reload, múltiplos workers ou upload feito antes do ask.
        self._ensure_case_index(case_id, documents)

        # 1) Evidências do case/upload atual SEMPRE entram quando houver documentos.
        top_k = 16 if focus in {"case_upload_first", "hybrid_case_first"} else 10
        case_evidences: list[dict[str, Any]] = []
        if documents:
            try:
                case_evidences = self.retrieval_service.search(
                    case_id=case_id,
                    query=question,
                    top_k=top_k,
                    documents=documents,
                    focus=focus,
                )
            except TypeError:
                case_evidences = self.retrieval_service.search(case_id, question, top_k=top_k)
            case_evidences = self._tag_evidences(case_evidences, default_source="case_upload")

        # 2) Consulta tabular como complemento. Nunca deve abafar arquivo/anexo.
        tabular_result = self._answer_tabular(case_id, question, documents, mode, focus)
        tabular_evidences = self._extract_tabular_evidences(tabular_result)

        # 3) Mescla as fontes na ordem correta.
        evidences = self._merge_evidences(
            case_evidences=case_evidences,
            tabular_evidences=tabular_evidences,
            focus=focus,
        )

        # 4) Resposta tabular direta só quando for seguro.
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
                    "evidence_count": len(evidences),
                }
                if tabular_result:
                    formatted["tabular_attempt"] = tabular_result.get("technical")
                return formatted

        # Fallback sem LLM.
        if tabular_result and not tabular_result.get("fallback_to_rag") and not case_evidences:
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
            "evidence_count": len(evidences),
        }
        if tabular_result:
            formatted["tabular_attempt"] = tabular_result.get("technical")
        return formatted

    def _ensure_case_index(self, case_id: str, documents: list[dict[str, Any]]) -> None:
        if not documents:
            return
        try:
            status = self.retrieval_service.status()
            indexed = False
            if hasattr(self.retrieval_service, "has_case_index"):
                indexed = bool(self.retrieval_service.has_case_index(case_id))
            else:
                indexed = case_id in getattr(self.retrieval_service, "case_chunks", {})
            if not indexed:
                self.retrieval_service.build_case_index(case_id, documents)
        except Exception:
            # Não bloqueia a resposta; o fallback tentará responder com o que houver.
            pass

    def _answer_tabular(
        self,
        case_id: str,
        question: str,
        documents: list[dict[str, Any]],
        mode: str,
        focus: str,
    ) -> dict[str, Any] | None:
        try:
            return self.tabular_service.answer_question(
                case_id=case_id,
                question=question,
                documents=documents,
                mode=mode,
                source_preference=focus,
            )
        except TypeError:
            return self.tabular_service.answer_question(case_id, question, documents, mode=mode)
        except Exception:
            return None

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
        file_focus = bool(tokens & self.FILE_FOCUS_TERMS) or any(term in q for term in self.FILE_FOCUS_TERMS if " " in term)
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

        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for e in merged:
            key = (str(e.get("filename", "")), str(e.get("chunk_id", "")), str(e.get("type", "")))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out[:20]

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
        if focus in {"case_upload_first", "hybrid_case_first"} and case_evidences:
            return False
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
        )[:20000]

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
2. Quando houver arquivo/documento/planilha/PDF/TXT enviado no front, trate esse conteúdo como contexto prioritário do case.
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
