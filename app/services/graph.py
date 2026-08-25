from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService
from app.services.knowledge_structured_store import KnowledgeStructuredStore


class AnalysisGraphService:
    """
    Orquestrador principal do chat do Nexus.

    Correção aplicada nesta versão:
    - botão único de conversa continua suportando documento, base treinada e híbrido;
    - perguntas documentais abertas passam a ter prioridade real sobre base/Postgres;
    - base treinada não substitui documento anexado quando a pergunta fala do arquivo/material;
    - perguntas analíticas estruturadas continuam determinísticas;
    - fallback sem LLM deixa de responder genericamente sobre automação quando a intenção é documental.
    """

    FILE_FOCUS_TERMS = {
        "arquivo", "arquivos", "anexo", "anexos", "anexado", "anexada", "anexados", "anexadas",
        "documento", "documentos", "doc", "docs", "pdf", "docx", "txt", "csv", "xlsx", "xls",
        "planilha", "excel", "upload", "uploads", "enviado", "enviada", "enviados", "enviadas",
        "material", "materiais", "contrato", "minuta", "termo", "relatorio", "relatório", "arquivo enviado",
        "neste", "nessa", "nesta", "nesse", "desse", "deste", "dessa", "desta", "dele", "nele", "nela",
    }

    KNOWLEDGE_FOCUS_TERMS = {
        "base", "conhecimento", "treinamento", "treinamentos", "treinado", "treinada",
        "article", "artigo", "artigos", "topic", "topico", "tópico", "historico", "histórico",
        "memoria", "memória", "gabbi", "nexus", "kb", "base treinada", "base de conhecimento",
        "knowledge", "knowledge base",
    }

    # Vocabulário amplo para detectar perguntas abertas sobre o conteúdo do documento.
    DOCUMENT_INTENT_TERMS = {
        # resumo / teor / assunto
        "resuma", "resumo", "sumarize", "sumariza", "sumarizar", "sintetize", "sintese", "síntese",
        "explique", "explica", "explicar", "descreva", "descrever", "detalhe", "detalhar",
        "teor", "conteudo", "conteúdo", "assunto", "tema", "tematica", "temática", "contexto",
        "do que se trata", "sobre o que", "qual e o teor", "qual é o teor", "qual o teor",
        "qual e o assunto", "qual é o assunto", "qual o assunto", "o que diz", "o que fala",
        "o que consta", "o que contem", "o que contém", "entenda", "interpretar", "interprete",

        # documento/contrato/minuta
        "documento", "arquivo", "anexo", "pdf", "material", "contrato", "minuta", "termo",
        "clausula", "cláusula", "clausulas", "cláusulas", "objeto", "escopo", "vigencia", "vigência",
        "partes", "contratante", "contratada", "fornecedor", "cliente", "prestacao", "prestação",
        "servicos", "serviços", "obrigações", "obrigacoes", "responsabilidades", "penalidades",
        "multa", "prazo", "pagamento", "valor", "preco", "preço", "sla", "risco", "riscos",

        # análise documental
        "analise", "análise", "analise o documento", "análise do documento", "parecer", "avaliar", "avalie",
        "pontos principais", "principais pontos", "destaques", "pontos de atenção", "pontos de atencao",
        "riscos do documento", "riscos contratuais", "inconsistencias", "inconsistências",
    }

    DOCUMENT_INTENT_PHRASES = {
        "explique o teor", "explica o teor", "qual o teor", "qual é o teor", "qual e o teor",
        "explique o documento", "explica o documento", "resuma o documento", "resumo do documento",
        "do que se trata", "sobre o que é", "sobre o que e", "o que fala o documento",
        "o que diz o documento", "o que consta no documento", "o que contem no documento", "o que contém no documento",
        "explique esse documento", "explique este documento", "explique este arquivo", "explique esse arquivo",
        "resuma esse documento", "resuma este documento", "resuma esse arquivo", "resuma este arquivo",
        "analise esse documento", "analise este documento", "avalie esse documento", "avalie este documento",
        "quais os principais pontos", "pontos principais do documento", "pontos de atenção do documento",
        "riscos do contrato", "riscos do documento", "clausulas do contrato", "cláusulas do contrato",
    }

    DOCUMENT_REFERENCE_TERMS = {
        "documento", "documentos", "arquivo", "arquivos", "anexo", "anexos", "pdf", "docx", "txt", "csv", "xlsx",
        "planilha", "material", "contrato", "minuta", "termo", "relatorio", "relatório", "upload",
        "este", "esse", "esta", "essa", "isto", "isso", "nele", "nela", "dele", "dela",
    }

    DOCUMENT_ACTION_TERMS = {
        "explique", "explica", "resuma", "resumo", "sintetize", "descreva", "detalhe", "analise", "análise",
        "avalie", "interprete", "informe", "liste", "mostre", "aponte", "identifique", "extraia", "extração",
        "qual", "quais", "o que", "do que", "sobre",
    }

    def __init__(self, retrieval_service, analysis_service):
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.llm_service = LLMService()
        self.tabular_service = TabularQueryService(llm_service=self.llm_service)
        self.knowledge_structured_store = KnowledgeStructuredStore()
        self.force_rag_first = self._env_bool("GABBI_FORCE_RAG_FIRST", False)
        self.always_synthesize_hybrid = self._env_bool("GABBI_ALWAYS_SYNTHESIZE_HYBRID", False)

        # Orçamento de contexto documental. O corte deixa de ser um slice global
        # que eliminava silenciosamente os últimos documentos.
        self.document_context_max_chars = int(os.getenv("GABBI_DOCUMENT_CONTEXT_MAX_CHARS", "60000"))
        self.document_excerpt_max_chars = int(os.getenv("GABBI_DOCUMENT_EXCERPT_MAX_CHARS", "3500"))
        self.document_excerpt_min_chars = int(os.getenv("GABBI_DOCUMENT_EXCERPT_MIN_CHARS", "600"))
        self.case_wide_extra_chunks = int(os.getenv("GABBI_CASE_WIDE_EXTRA_CHUNKS", "8"))
        self.max_merged_evidences = int(os.getenv("GABBI_MAX_MERGED_EVIDENCES", "200"))
        self.document_adaptive_top_k = int(os.getenv("GABBI_DOCUMENT_ADAPTIVE_TOP_K", "20"))

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
        has_documents = bool(documents)
        # Architecture V5:
        # o Nexus é um analista documental. Se o case possui documentos, a busca
        # documental é válida independentemente do domínio/agente.
        is_document_intent = bool(documents)
        is_case_wide = False
        focus = "case_upload_first" if is_document_intent else self._detect_focus(question, documents)
        is_analytics = self._is_analytics_question(question)

        # 0. Analytics estruturado continua prioritário, mas somente quando for de fato pergunta analítica.
        # Para perguntas documentais abertas, não consultamos primeiro a base Postgres, para não substituir o upload.
        if is_analytics:
            try:
                structured_result = self.knowledge_structured_store.answer_question(
                    case_id=case_id,
                    question=question,
                    chat_history=chat_history or [],
                )
                if structured_result and not structured_result.get("fallback_to_rag"):
                    structured_result.setdefault("sources", {})
                    if isinstance(structured_result.get("sources"), dict):
                        structured_result["sources"].update({
                            "deterministic": True,
                            "llm_synthesis_blocked": True,
                            "reason": "structured_analytics_first",
                        })
                    return structured_result
            except Exception as exc:
                return self._deterministic_error(
                    case_id=case_id,
                    question=question,
                    route="knowledge_structured_error",
                    message=f"Erro ao consultar a engine estruturada: {type(exc).__name__}: {exc}",
                )

            tabular_result = self.tabular_service.answer_question(
                case_id=case_id,
                question=question,
                documents=documents,
                mode=mode,
                source_preference=focus,
            )
            if tabular_result and not tabular_result.get("fallback_to_rag"):
                tabular_result["route"] = "tabular_direct_no_llm"
                tabular_result["sources"] = {
                    "deterministic": True,
                    "llm_synthesis_blocked": True,
                    "reason": "analytics_question",
                    "focus": focus,
                    "document_intent": is_document_intent,
                    "tabular_source_type": self._tabular_source_type(tabular_result),
                }
                return tabular_result

            return self._deterministic_error(
                case_id=case_id,
                question=question,
                route="analytics_no_deterministic_result",
                message=(
                    "Não consegui calcular essa pergunta de forma determinística na base estruturada/tabular disponível. "
                    "A resposta foi bloqueada para evitar síntese por RAG/LLM com números incorretos. "
                    "Verifique se o case foi sincronizado/reindexado e se os campos necessários existem."
                ),
                technical={"tabular_attempt": tabular_result.get("technical") if tabular_result else None},
            )

        # 1. Evidências do upload/case atual.
        case_evidences: list[dict[str, Any]] = []
        if has_documents:
            if hasattr(self.retrieval_service, "ensure_case_index"):
                try:
                    self.retrieval_service.ensure_case_index(case_id, documents)
                except Exception:
                    pass
            if is_document_intent and hasattr(self.retrieval_service, "search_adaptive"):
                # Busca adaptativa e agnóstica:
                # - preserva semântica em perguntas focadas;
                # - preserva arquivos e regiões em documentos longos/cases amplos.
                profile = self.retrieval_service.coverage_profile(case_id, question)
                is_case_wide = bool(profile.get("recommended"))
                requested_top_k = max(
                    self.document_adaptive_top_k,
                    len(documents) + self.case_wide_extra_chunks,
                )
                case_evidences = self.retrieval_service.search_adaptive(
                    case_id=case_id,
                    query=question,
                    top_k=requested_top_k,
                    prefer_coverage=None,
                )
            else:
                top_k = 8 if is_document_intent else 6
                case_evidences = self.retrieval_service.search(case_id, question, top_k=top_k)

            case_evidences = self._tag_evidences(case_evidences, default_source="case_upload")

        # 2. Tabular como apoio em perguntas semânticas; para intenção documental só complementa, não retorna direto.
        tabular_result = self.tabular_service.answer_question(
            case_id=case_id,
            question=question,
            documents=documents,
            mode=mode,
            source_preference=focus,
        )
        tabular_evidences = [] if is_document_intent else self._extract_tabular_evidences(tabular_result)

        # 3. Mescla evidências com prioridade correta.
        evidences = self._merge_evidences(
            case_evidences=case_evidences,
            tabular_evidences=tabular_evidences,
            focus=focus,
            max_items=(
                min(
                    self.max_merged_evidences,
                    max(self.document_adaptive_top_k, len(case_evidences), len(documents) + self.case_wide_extra_chunks),
                )
                if is_document_intent
                else 16
            ),
        )

        # 4. Retorno tabular direto somente quando não for intenção documental.
        if (
            not is_document_intent
            and tabular_result
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
                tabular_attempt=None if is_document_intent else tabular_result,
                focus=focus,
                document_intent=is_document_intent,
                case_wide=is_case_wide,
            )
            if answer:
                formatted["summary"] = answer
                formatted["answer_text"] = answer
                formatted["route"] = "document" if is_document_intent else "hybrid"
                formatted["query_type"] = "document_qa" if is_document_intent else "hybrid_qa"
                formatted["sources"] = {
                    "focus": focus,
                    "document_intent": is_document_intent,
                    "case_wide": is_case_wide,
                    "case_uploads": bool(case_evidences),
                    "evidence_files": sorted({str(e.get("filename")) for e in case_evidences if e.get("filename")}),
                    "case_documents": [
                        str(doc.get("filename") or "arquivo")
                        for doc in documents
                    ],
                    "case_document_count": len(documents),
                    "evidence_count": len(case_evidences),
                    "coverage_roles": sorted({
                        str((e.get("metadata") or {}).get("coverage_role") or "semantic")
                        for e in case_evidences
                    }),
                    "coverage_buckets": sorted({
                        str((e.get("metadata") or {}).get("coverage_bucket"))
                        for e in case_evidences
                        if (e.get("metadata") or {}).get("coverage_bucket")
                    }),
                    "tabular": bool(tabular_result and not tabular_result.get("fallback_to_rag") and not is_document_intent),
                    "tabular_source_type": self._tabular_source_type(tabular_result),
                    "conversation_memory": bool(history),
                }
                if tabular_result and not is_document_intent:
                    formatted["tabular_attempt"] = tabular_result.get("technical")
                return formatted

        # 5. Fallback sem LLM: se era intenção documental, não usar texto genérico de automação.
        if is_document_intent:
            fallback = self._fallback_document_answer(question, evidences, analysis, mode=mode)
            formatted.update(fallback)
            return formatted

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
            "document_intent": is_document_intent,
            "case_uploads": bool(case_evidences),
            "tabular": bool(tabular_result),
            "conversation_memory": bool(history),
        }
        if tabular_result:
            formatted["tabular_attempt"] = tabular_result.get("technical")
        return formatted

    def _deterministic_error(
        self,
        case_id: str,
        question: str,
        route: str,
        message: str,
        technical: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "fallback_to_rag": False,
            "route": route,
            "query_type": "analytics_blocked_no_deterministic_result",
            "answer_text": message,
            "summary": message,
            "technical": {"case_id": case_id, "question": question, **(technical or {})},
            "sources": {"deterministic": True, "llm_synthesis_blocked": True, "rag_blocked": True},
        }

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
        file_focus = bool(tokens & self._norm_set(self.FILE_FOCUS_TERMS))
        kb_focus = bool(tokens & self._norm_set(self.KNOWLEDGE_FOCUS_TERMS))

        if file_focus and has_docs:
            return "case_upload_first"
        if kb_focus and not file_focus:
            return "knowledge_base_first"
        if has_docs:
            return "hybrid_case_first"
        return "knowledge_base_first"

    def _is_document_intent(self, question: str, documents: list[dict[str, Any]]) -> bool:
        if not documents:
            return False

        q = self._norm(question)
        tokens = set(q.split())
        normalized_phrases = self._norm_set(self.DOCUMENT_INTENT_PHRASES)
        normalized_doc_terms = self._norm_set(self.DOCUMENT_INTENT_TERMS | self.DOCUMENT_REFERENCE_TERMS)
        normalized_ref_terms = self._norm_set(self.DOCUMENT_REFERENCE_TERMS)
        normalized_action_terms = self._norm_set(self.DOCUMENT_ACTION_TERMS)

        # Frases fortes: "explique o teor", "do que se trata", "resuma o documento".
        if any(phrase in q for phrase in normalized_phrases):
            return True

        # Combinação ação + referência documental.
        if (tokens & normalized_action_terms) and (tokens & normalized_ref_terms):
            return True

        # Termos documentais fortes em perguntas curtas.
        if tokens & normalized_doc_terms:
            question_markers = {"qual", "quais", "o", "que", "como", "explique", "resuma", "analise", "avalie", "liste", "mostre"}
            if tokens & question_markers:
                return True

        # Continuação conversacional: "e os riscos?", "e o valor?" após upload/documento.
        continuation_terms = {
            "risco", "riscos", "valor", "prazo", "objeto", "escopo", "partes", "obrigacoes", "obrigações",
            "multa", "penalidade", "penalidades", "pagamento", "vigencia", "vigência", "clausula", "cláusula",
            "contratante", "contratada", "responsabilidade", "responsabilidades", "sla",
        }
        if len(tokens) <= 8 and bool(tokens & self._norm_set(continuation_terms)):
            return True

        return False

    def _is_case_wide_document_question(self, question: str) -> bool:
        """Detecta pedidos que exigem cobertura do case completo, não só TOP-K semântico."""
        q = self._norm(question)

        strong_markers = [
            "todos os documentos", "todos os arquivos", "todos os anexos",
            "documentacao completa", "documentação completa",
            "analise os documentos", "analise todos os documentos", "analise todos os arquivos",
            "analise completa", "análise completa",
            "triagem completa", "faca a triagem", "faça a triagem",
            "consolide os documentos", "consolidar os documentos",
            "considere todos", "considerando todos", "considerar todos",
            "dossie completo", "dossiê completo", "todo o dossie", "todo o dossiê",
            "conjunto completo", "pacote completo",
            "cruze os documentos", "cruzar os documentos", "conferencia cruzada", "conferência cruzada",
        ]
        if any(marker in q for marker in self._norm_set(set(strong_markers))):
            return True

        # Combina verbos globais com referências plurais a documentos.
        global_actions = {"analise", "avalie", "consolide", "compare", "cruze", "triagem", "verifique"}
        plural_refs = {"documentos", "arquivos", "anexos", "materiais", "comprovantes", "laudos"}
        tokens = set(q.split())
        return bool(tokens & global_actions and tokens & plural_refs)

    def _merge_evidences(
        self,
        case_evidences: list[dict[str, Any]],
        tabular_evidences: list[dict[str, Any]],
        focus: str,
        max_items: int | None = None,
    ) -> list[dict[str, Any]]:
        case = self._tag_evidences(case_evidences, "case_upload")
        tabular = tabular_evidences or []
        merged = case + tabular if focus in {"case_upload_first", "hybrid_case_first"} else tabular + case
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for e in merged:
            key = (str(e.get("filename", "")), str(e.get("chunk_id", "")), str(e.get("type", "")))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        limit = max(1, int(max_items or 16))
        return out[:limit]

    def _tag_evidences(self, evidences: list[dict[str, Any]], default_source: str) -> list[dict[str, Any]]:
        out = []
        for item in evidences or []:
            cloned = dict(item)
            metadata = dict(cloned.get("metadata") or {})
            metadata.setdefault("source", default_source)
            metadata.setdefault("source_type", default_source)
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
        return [{
            "filename": filename,
            "chunk_id": "tabular_execution",
            "type": "tabular",
            "score": 1.0,
            "excerpt": json.dumps(excerpt_payload, ensure_ascii=False, default=str)[:5000],
            "metadata": {"source": "tabular", "source_type": source_type},
        }]

    def _can_return_tabular_direct(self, tabular_result: dict[str, Any], case_evidences: list[dict[str, Any]], focus: str) -> bool:
        source_type = self._tabular_source_type(tabular_result)
        if source_type == "file":
            return True
        if source_type == "postgres" and not case_evidences:
            return True
        if focus == "knowledge_base_first":
            return True
        return False

    def _is_analytics_question(self, question: str) -> bool:
        """Detecta intenção analítica sem vocabulário de domínio.

        Só considera analytics quando houver operador quantitativo/estrutural forte.
        Termos genéricos como "quais" ou "liste" sozinhos NÃO bloqueiam RAG.
        """
        q = self._norm(question)
        strong_patterns = [
            "quantos", "quantas", "quantidade", "contagem", "conte",
            "total de", "soma de", "somatorio", "somatório",
            "media de", "média de", "mediana", "percentual", "porcentagem",
            "agrupe por", "agrupar por", "distribuicao por", "distribuição por",
            "ranking", "top ", "maior quantidade", "menor quantidade",
            "distintos", "distintas", "distinct",
            "por mes", "por mês", "por dia", "por ano", "por status",
        ]
        return any(pattern in q for pattern in strong_patterns)

    def _tabular_source_type(self, tabular_result: dict[str, Any] | None) -> str | None:
        if not tabular_result:
            return None
        execution = ((tabular_result.get("technical") or {}).get("execution") or {})
        table = execution.get("table") or {}
        return table.get("source_type")

    def _build_evidence_blob(
        self,
        evidences: list[dict[str, Any]],
        *,
        case_wide: bool = False,
    ) -> str:
        """Monta contexto por unidades de cobertura, não por nome de arquivo.

        V4 distribuía orçamento por arquivo. Isso falhava quando um único PDF
        continha muitas seções/documentos lógicos. V5 distribui por evidência/
        região estrutural selecionada, preservando início/meio/fim e relevância.
        """
        if not evidences:
            return ""

        max_context = max(4000, self.document_context_max_chars)
        max_excerpt = max(400, self.document_excerpt_max_chars)
        min_excerpt = max(250, min(self.document_excerpt_min_chars, max_excerpt))

        # Unidades são (arquivo + coverage_bucket/role/chunk), portanto um único
        # PDF grande pode ocupar várias regiões de contexto sem ser truncado a
        # um único budget de arquivo.
        units = []
        seen = set()
        for evidence in evidences:
            metadata = evidence.get("metadata") or {}
            filename = str(evidence.get("filename") or "fonte")
            bucket = str(metadata.get("coverage_bucket") or "")
            role = str(metadata.get("coverage_role") or "semantic")
            chunk_id = str(evidence.get("chunk_id") or "")
            unit_key = (filename, bucket or role, chunk_id)
            if unit_key in seen:
                continue
            seen.add(unit_key)
            units.append(evidence)

        if not units:
            return ""

        overhead = 190 * len(units)
        available = max(min_excerpt * len(units), max_context - overhead)
        per_unit_budget = max(
            min_excerpt,
            min(max_excerpt, available // max(1, len(units))),
        )

        blocks: list[str] = []
        used_chars = 0

        for evidence in units:
            metadata = evidence.get("metadata") or {}
            filename = str(evidence.get("filename") or "fonte")
            source = metadata.get("source") or metadata.get("source_type")
            role = metadata.get("coverage_role") or "semantic"
            bucket = metadata.get("coverage_bucket")
            excerpt = str(evidence.get("excerpt") or "").strip()
            if not excerpt:
                continue

            excerpt = excerpt[:per_unit_budget]
            header = (
                f"[{filename} | chunk={evidence.get('chunk_id')} | "
                f"regiao={bucket or '-'} | papel={role} | "
                f"score={evidence.get('score')} | fonte={source}]\n"
            )
            block = header + excerpt

            if used_chars + len(block) > max_context:
                remaining = max_context - used_chars
                if remaining <= len(header) + min_excerpt:
                    break
                block = header + excerpt[:max(0, remaining - len(header))]

            blocks.append(block)
            used_chars += len(block) + 2

            if used_chars >= max_context:
                break

        return "\n\n".join(blocks)

    def _ask_openai(
        self,
        question: str,
        analysis: dict[str, Any],
        evidences: list[dict[str, Any]],
        history: list[dict[str, str]],
        mode: str,
        tabular_attempt: dict[str, Any] | None = None,
        focus: str = "hybrid_case_first",
        document_intent: bool = False,
        case_wide: bool = False,
    ) -> str | None:
        evidence_blob = self._build_evidence_blob(evidences, case_wide=case_wide)

        tabular_blob = ""
        if tabular_attempt:
            tabular_blob = "\n\nResultado/diagnóstico tabular, quando houver. Use apenas se estiver coerente com as evidências do upload e da base:\n"
            tabular_blob += json.dumps(tabular_attempt.get("technical", tabular_attempt), ensure_ascii=False, indent=2, default=str)[:7000]

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

        if document_intent:
            system_prompt += """
REGRAS UNIVERSAIS DE EVIDÊNCIA DOCUMENTAL:
- O inventário de arquivos recebidos e o conteúdo efetivamente recuperado são conceitos diferentes.
- Nunca conclua que um documento/artefato está ausente apenas porque um trecho não apareceu nas evidências selecionadas.
- Quando o arquivo foi recebido, mas uma informação esperada não foi recuperada, descreva como limitação de leitura/recuperação técnica.
- Ausência só pode ser afirmada quando houver evidência positiva de que o item não foi fornecido ou não existe no material analisado.
- Não complete lacunas de um documento por analogia com outro documento.
- Preserve datas, números, códigos, nomes e valores exatamente como aparecem nas evidências.

MODO DOCUMENTAL OBRIGATÓRIO:
- A pergunta foi classificada como pergunta sobre o documento/anexo atual.
- Responda diretamente sobre o conteúdo do documento anexado.
- Não responda com recomendação genérica de automação, RAG, classificação documental ou extração de campos, salvo se o usuário pedir isso explicitamente.
- Para "explique o teor", "resuma", "do que se trata" ou perguntas similares, explique: natureza do documento, partes envolvidas, objeto/escopo, valores/prazos, obrigações, penalidades/riscos e conclusão executiva, somente quando essas informações aparecerem nas evidências.
- Se as evidências forem trechos parciais, deixe claro que o resumo está baseado nos trechos recuperados.
- Em análise de case completo, não declare um arquivo como ausente/ilegível apenas porque um campo não apareceu em outro documento. Use o nome da fonte e o trecho efetivamente recebido.
- Quando houver múltiplos documentos, faça conferências cruzadas somente com dados presentes nas evidências; não invente ausência.
""".rstrip()

        if mode == "executive":
            system_prompt += "\nPriorize linguagem executiva, objetiva e orientada à decisão."
        elif mode == "analytical":
            system_prompt += "\nPriorize análise detalhada, riscos, regras, exceções e automação recomendada quando aplicável."
        else:
            system_prompt += "\nPriorize precisão técnica."

        user_prompt = f"""
Foco de roteamento detectado: {focus}
Intenção documental detectada: {document_intent}
Análise de case completo detectada: {case_wide}

Pergunta do usuário:
{question}

Contexto analítico do caso:
{json.dumps(analysis, ensure_ascii=False, indent=2, default=str)[:8000]}

Evidências recuperadas pelo Nexus:
{evidence_blob}
{tabular_blob}

Gere uma resposta organizada em markdown.
"""
        return self.llm_service.generate_chat(system_prompt, history[-8:], user_prompt)

    def _fallback_document_answer(
        self,
        question: str,
        evidences: list[dict[str, Any]],
        analysis: dict[str, Any],
        mode: str = "executive",
    ) -> dict[str, Any]:
        files = sorted({str(e.get("filename")) for e in evidences if e.get("filename")})
        excerpts = []
        for e in evidences[:4]:
            excerpt = str(e.get("excerpt") or "").strip()
            if excerpt:
                excerpts.append(excerpt[:700])
        if excerpts:
            summary = (
                "Identifiquei conteúdo do documento anexado, mas o LLM não está habilitado ou não retornou resposta. "
                "Com base nos trechos recuperados, o material deve ser analisado a partir das evidências abaixo."
            )
        else:
            summary = (
                "Não encontrei evidências suficientes do documento anexado para responder com segurança. "
                "Verifique se o upload foi processado e se o índice do case foi reconstruído."
            )
        answer_text = summary
        if files:
            answer_text += "\n\n**Arquivo(s) considerado(s):** " + ", ".join(files)
        if excerpts:
            answer_text += "\n\n**Trechos recuperados:**\n" + "\n\n".join(f"- {x}" for x in excerpts)
        return {
            "summary": answer_text,
            "answer_text": answer_text,
            "route": "document",
            "query_type": "document_qa",
            "sources": {
                "focus": "case_upload_first",
                "document_intent": True,
                "case_uploads": bool(evidences),
                "llm_enabled": bool(self.llm_service.status().get("enabled")),
            },
        }

    @staticmethod
    def _strip_accents(value: str) -> str:
        return "".join(
            ch for ch in unicodedata.normalize("NFD", value)
            if unicodedata.category(ch) != "Mn"
        )

    @classmethod
    def _norm_set(cls, values: set[str]) -> set[str]:
        return {cls._norm(v) for v in values if cls._norm(v)}

    @staticmethod
    def _norm(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.lower().strip()
        text = "".join(ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn")
        text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
        return " ".join(text.split())
