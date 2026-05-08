from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService

try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except Exception:
    HAS_SQLALCHEMY = False


class AnalysisGraphService:
    """Orquestrador do chat do Nexus.

    Versão Enterprise Exact Retrieval:
    - Perguntas quantitativas/listagens não dependem de top_k/RAG.
    - CSV/XLSX anexado é lido por conteúdo completo quando o parser disponibiliza texto/tabelas.
    - Perguntas sobre transações, CHG e INC usam varredura determinística antes da LLM.
    - Para base Gabbi, consulta PostgreSQL direto quando DATABASE_URL/GABBI_DATABASE_URL estiver configurado.
    - LLM só sintetiza quando não há resposta determinística direta.
    """

    FILE_FOCUS_TERMS = {
        "arquivo", "arquivos", "anexo", "anexos", "anexado", "anexados", "documento", "documentos",
        "planilha", "excel", "pdf", "docx", "csv", "txt", "upload", "uploads", "enviado", "enviados",
        "teor", "escopo", "conteudo", "conteúdo", "material"
    }
    COUNT_TERMS = {
        "quantos", "quantas", "quantidade", "qtd", "qtde", "total", "totais", "contar", "conte",
        "número", "numero", "somar", "some", "soma"
    }
    LIST_TERMS = {"liste", "listar", "lista", "quais", "mostre", "exiba", "relacione", "detalhe", "detalhar"}
    KNOWLEDGE_TERMS = {
        "base", "conhecimento", "treinamento", "treinado", "treinada", "artigo", "artigos", "topic", "tópico", "topico",
        "banco", "postgres", "dados"
    }

    def __init__(self, retrieval_service, analysis_service):
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.llm_service = LLMService()
        self.tabular_service = TabularQueryService(llm_service=self.llm_service)
        self.max_direct_file_chars = int(os.getenv("GABBI_DIRECT_FILE_CONTEXT_CHARS", "40000"))
        self.max_scan_records = int(os.getenv("GABBI_DETERMINISTIC_SCAN_MAX_RECORDS", "200000"))
        self.db_url = (
            os.getenv("GABBI_DATABASE_URL", "").strip()
            or os.getenv("GABBI_POSTGRES_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )
        self.db_max_rows = int(os.getenv("GABBI_ANALYTICS_DB_MAX_ROWS", "200000"))
        self._db_engine = None
        if HAS_SQLALCHEMY and self.db_url:
            try:
                self._db_engine = create_engine(self.db_url, future=True, pool_pre_ping=True)
            except Exception:
                self._db_engine = None

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
        self._ensure_case_index(case_id, documents)
        focus = self._detect_focus(question, documents)
        question_with_context = self._question_with_recent_context(question, history)

        # 1) Primeiro: engine determinística. Se ela responde, a LLM não pode recontar nem aproximar.
        deterministic = self._deterministic_answer(
            case_id=case_id,
            question=question,
            question_with_context=question_with_context,
            documents=documents,
            history=history,
            focus=focus,
        )
        if deterministic:
            return deterministic

        # 2) Evidência direta dos arquivos anexados.
        file_evidences = self._direct_file_evidences(question, documents, focus)

        # 3) RAG apenas para perguntas descritivas, explicativas ou quando não há resposta exata.
        top_k = 32 if focus in {"case_upload_first", "hybrid_case_first"} else 24
        case_evidences = []
        if documents:
            case_evidences = self.retrieval_service.search(case_id, question, top_k=top_k)
            case_evidences = self._tag_evidences(case_evidences, "case_upload_or_knowledge_in_case")

        # 4) Tabular como complemento; nunca deixa row_count substituir contagem interna de entidades.
        tabular_result = None
        try:
            tabular_result = self.tabular_service.answer_question(
                case_id=case_id,
                question=question,
                documents=documents,
                mode=mode,
                source_preference=focus,
            )
        except Exception as exc:
            tabular_result = {"fallback_to_rag": True, "technical": {"error": str(exc)}}

        tabular_evidences = self._extract_tabular_evidences(tabular_result)
        evidences = self._merge_evidences(file_evidences, case_evidences, tabular_evidences, focus)

        formatted = self.analysis_service.format_answer(question, evidences, analysis, mode=mode)

        if self.llm_service.status()["enabled"]:
            answer = self._ask_openai(question, analysis, evidences, history, mode, tabular_result, focus)
            if answer:
                formatted.update({
                    "summary": answer,
                    "answer_text": answer,
                    "route": "hybrid_exact_guarded",
                    "query_type": "document_qa",
                    "sources": {
                        "focus": focus,
                        "documents_in_case": len(documents or []),
                        "direct_file_evidence": bool(file_evidences),
                        "rag_evidences": len(case_evidences),
                        "tabular": bool(tabular_result and not tabular_result.get("fallback_to_rag")),
                        "conversation_memory": bool(history),
                    },
                })
                if tabular_result:
                    formatted["tabular_attempt"] = tabular_result.get("technical")
                return formatted

        if evidences:
            answer = self._fallback_evidence_answer(question, evidences)
            formatted.update({"summary": answer, "answer_text": answer, "route": "evidence_fallback", "query_type": "document_qa"})
            return formatted

        msg = "Não encontrei evidência textual suficiente nos documentos indexados deste case para responder com precisão."
        formatted.update({"answer_text": msg, "summary": msg, "route": "no_evidence", "query_type": "document_qa"})
        return formatted

    # ------------------------------------------------------------------
    # Deterministic engine
    # ------------------------------------------------------------------

    def _deterministic_answer(
        self,
        *,
        case_id: str,
        question: str,
        question_with_context: str,
        documents: list[dict[str, Any]],
        history: list[dict[str, str]],
        focus: str,
    ) -> dict[str, Any] | None:
        q_norm = self._norm(question)
        qc_norm = self._norm(question_with_context)
        tokens = set(q_norm.split())
        is_count = bool(tokens & self.COUNT_TERMS)
        is_list = bool(tokens & self.LIST_TERMS)
        if not (is_count or is_list):
            return None

        # 1. Arquivo anexado: contagem de transações internas, soma e linhas.
        file_answer = self._deterministic_file_answer(case_id, question, documents, focus, is_count=is_count, is_list=is_list)
        if file_answer:
            return file_answer

        # 2. Base Gabbi/PostgreSQL: CHG/INC com consulta direta aos artigos.
        wants_chg = self._wants_entity(qc_norm, "chg")
        wants_inc = self._wants_entity(qc_norm, "inc")
        if wants_chg or wants_inc:
            entity = "CHG" if wants_chg else "INC"
            db_answer = self._deterministic_gabbi_db_answer(case_id, question, question_with_context, entity, is_count=is_count, is_list=is_list)
            if db_answer:
                return db_answer

            # Fallback sem DB: varrer todos os documentos do case.
            scan_answer = self._deterministic_code_scan_answer(case_id, question, question_with_context, documents, entity, is_count=is_count, is_list=is_list)
            if scan_answer:
                return scan_answer

        return None

    def _deterministic_file_answer(self, case_id: str, question: str, documents: list[dict[str, Any]], focus: str, is_count: bool, is_list: bool) -> dict[str, Any] | None:
        q_norm = self._norm(question)
        # Só entra para pergunta de arquivo/anexo ou quando há um único upload tabular/textual no case.
        file_docs = [d for d in documents or [] if self._document_source_type(d) == "case_upload"]
        if not file_docs:
            return None
        if focus != "case_upload_first" and len(file_docs) != 1 and not any(t in q_norm for t in self.FILE_FOCUS_TERMS):
            return None

        combined_text = "\n\n".join(((d.get("parsed") or {}).get("text") or "") for d in file_docs)
        if not combined_text.strip():
            return None

        # Transações SAP: conta ocorrências reais de TRANSACAO: e deduplica se pedido.
        if any(t in q_norm for t in ["transacao", "transacoes", "transação", "transações"]):
            codes = self._extract_transaction_codes(combined_text)
            if not codes:
                return None
            unique = sorted(set(codes))
            distinct = any(t in q_norm for t in ["distint", "unica", "única", "unicas", "únicas", "diferente"])
            count = len(unique) if distinct else len(codes)
            title = "transações distintas" if distinct else "transações totais"
            body = f"{count}"
            return self._direct_answer(
                case_id=case_id,
                answer=body,
                route="deterministic_file_full_scan",
                query_type="count",
                technical={
                    "entity": "TRANSACAO",
                    "count_type": title,
                    "occurrences": len(codes),
                    "distinct": len(unique),
                    "files": [d.get("filename") for d in file_docs],
                    "note": "Contagem feita por varredura completa do texto extraído do arquivo, usando o padrão TRANSACAO: <codigo>.",
                },
            )

        # Soma de valores numéricos em tabelas CSV/XLSX.
        if any(t in q_norm for t in ["some", "soma", "somar", "valor", "valores"]):
            total = self._sum_numeric_column_from_documents(file_docs, question)
            if total is not None:
                txt = str(int(total)) if float(total).is_integer() else str(total)
                return self._direct_answer(case_id, txt, "deterministic_file_table_sum", "sum", {"total": total})

        # Registros/linhas do arquivo: usa metadata rows/tables, não LLM.
        if any(t in q_norm for t in ["registro", "registros", "linha", "linhas"]):
            total_rows = self._count_rows_from_documents(file_docs)
            if total_rows is not None:
                return self._direct_answer(case_id, str(total_rows), "deterministic_file_table_count", "count", {"rows": total_rows})

        return None

    def _deterministic_gabbi_db_answer(self, case_id: str, question: str, question_with_context: str, entity: str, is_count: bool, is_list: bool) -> dict[str, Any] | None:
        if self._db_engine is None:
            return None
        criteria = self._extract_structured_criteria(question_with_context, entity)
        try:
            rows = self._load_gabbi_articles_from_db(entity=entity, criteria=criteria)
        except Exception as exc:
            return None
        if not rows:
            return self._direct_answer(
                case_id,
                "0",
                "deterministic_gabbi_postgres",
                "count" if is_count else "list",
                {"entity": entity, "criteria": criteria, "rows_loaded": 0},
            )

        code_records = self._extract_code_records_from_rows(rows, entity)
        filtered = self._filter_code_records(code_records, criteria)
        codes = sorted({r["code"] for r in filtered})
        if is_count:
            answer = str(len(codes))
        else:
            answer = "\n".join(codes) if codes else "Nenhum registro encontrado."
        return self._direct_answer(
            case_id=case_id,
            answer=answer,
            route="deterministic_gabbi_postgres",
            query_type="count" if is_count else "list",
            technical={
                "entity": entity,
                "criteria": criteria,
                "articles_loaded": len(rows),
                "records_found_before_filter": len(code_records),
                "records_found_after_filter": len(filtered),
                "distinct_codes": len(codes),
                "source": "public.Article/public.Topic/public.Project",
                "note": "Consulta determinística direta ao PostgreSQL do Gabbi; a LLM não fez contagem.",
            },
        )

    def _deterministic_code_scan_answer(self, case_id: str, question: str, question_with_context: str, documents: list[dict[str, Any]], entity: str, is_count: bool, is_list: bool) -> dict[str, Any] | None:
        criteria = self._extract_structured_criteria(question_with_context, entity)
        rows = []
        for doc in documents or []:
            text_value = ((doc.get("parsed") or {}).get("text") or "").strip()
            if not text_value:
                continue
            rows.append({
                "article_id": doc.get("id"),
                "article_ref_id": None,
                "topic_name": "",
                "topic_description": "",
                "project_name": "",
                "article_text": text_value,
                "filename": doc.get("filename"),
            })
        if not rows:
            return None
        code_records = self._extract_code_records_from_rows(rows, entity)
        filtered = self._filter_code_records(code_records, criteria)
        codes = sorted({r["code"] for r in filtered})
        if not codes and criteria.get("has_specific"):
            return self._direct_answer(case_id, "0", "deterministic_case_full_scan", "count" if is_count else "list", {"entity": entity, "criteria": criteria})
        if not codes:
            return None
        answer = str(len(codes)) if is_count else "\n".join(codes)
        return self._direct_answer(
            case_id,
            answer,
            "deterministic_case_full_scan",
            "count" if is_count else "list",
            {"entity": entity, "criteria": criteria, "records_found": len(filtered), "distinct_codes": len(codes)},
        )

    def _load_gabbi_articles_from_db(self, *, entity: str, criteria: dict[str, Any]) -> list[dict[str, Any]]:
        entity_pattern = f"%{entity}%"
        where = [
            "coalesce(A.deleted, false) = false",
            "coalesce(A.published, false) = true",
            "coalesce(T.deleted, false) = false",
            "coalesce(T.active, true) = true",
            "coalesce(P.deleted, false) = false",
            "(A.article ILIKE :entity OR T.name ILIKE :entity OR T.description ILIKE :entity)",
        ]
        params: dict[str, Any] = {"entity": entity_pattern, "limit": self.db_max_rows}
        # Não filtra demais no SQL; carrega candidatos e filtra em Python para suportar variações de texto.
        sql = text(f'''
            SELECT
                A.id AS article_id,
                A."refId" AS article_ref_id,
                A.article AS article_text,
                T.id AS topic_id,
                T."refId" AS topic_ref_id,
                T.name AS topic_name,
                T.description AS topic_description,
                T."projectId" AS project_id,
                P.name AS project_name
            FROM public."Article" A
            INNER JOIN public."Topic" T ON T.id = A."topicId"
            INNER JOIN public."Project" P ON P.id = T."projectId"
            WHERE {' AND '.join(where)}
            ORDER BY T."refId", A."refId"
            LIMIT :limit
        ''')
        with self._db_engine.connect() as conn:
            result = conn.execute(sql, params)
            return [dict(row._mapping) for row in result]

    def _extract_code_records_from_rows(self, rows: list[dict[str, Any]], entity: str) -> list[dict[str, Any]]:
        pattern = re.compile(rf"\b{re.escape(entity)}\d{{5,}}\b", flags=re.IGNORECASE)
        out: list[dict[str, Any]] = []
        for row in rows:
            text_value = "\n".join([
                str(row.get("topic_name") or ""),
                str(row.get("topic_description") or ""),
                str(row.get("article_text") or ""),
            ])
            codes = sorted({m.group(0).upper() for m in pattern.finditer(text_value)})
            if not codes:
                continue
            record_texts = self._split_text_into_records(text_value)
            for code in codes:
                # tenta associar código ao menor bloco onde ele aparece; se não achar, usa artigo inteiro
                block = next((b for b in record_texts if code.lower() in b.lower()), text_value)
                out.append({
                    "code": code,
                    "text": block,
                    "article_id": row.get("article_id"),
                    "article_ref_id": row.get("article_ref_id"),
                    "topic_name": row.get("topic_name") or "",
                    "topic_description": row.get("topic_description") or "",
                    "project_name": row.get("project_name") or "",
                    "filename": row.get("filename") or "PostgreSQL:Article",
                })
        return out

    def _filter_code_records(self, records: list[dict[str, Any]], criteria: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for rec in records:
            haystack = self._norm("\n".join([
                rec.get("text", ""), rec.get("topic_name", ""), rec.get("topic_description", ""), rec.get("project_name", "")
            ]))
            ok = True
            month = criteria.get("month")
            if month:
                yyyy, mm = month.split("-")
                variants = [month, f"{mm}-{yyyy}", f"{mm}/{yyyy}", f"{criteria.get('entity','')}:{mm}-{yyyy}".lower()]
                if not any(self._norm(v) in haystack for v in variants):
                    ok = False
            topic = criteria.get("topic")
            if ok and topic and self._norm(topic) not in haystack:
                ok = False
            group = criteria.get("grupo_atribuicao")
            if ok and group and self._norm(group) not in haystack:
                ok = False
            if ok:
                out.append(rec)
        return out

    def _extract_structured_criteria(self, text_value: str, entity: str) -> dict[str, Any]:
        raw = text_value or ""
        criteria: dict[str, Any] = {"entity": entity, "has_specific": False}
        # topic CHG:10-2025
        m_topic = re.search(rf"\b{entity}\s*:\s*(0?[1-9]|1[0-2])[-/](20\d{{2}})\b", raw, flags=re.IGNORECASE)
        if m_topic:
            mm, yyyy = m_topic.groups()
            criteria["topic"] = f"{entity}:{mm.zfill(2)}-{yyyy}"
            criteria["month"] = f"{yyyy}-{mm.zfill(2)}"
            criteria["has_specific"] = True
        else:
            # month 10-2025 or 2025-10
            m = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", raw)
            if m:
                mm, yyyy = m.groups()
                criteria["month"] = f"{yyyy}-{mm.zfill(2)}"
                criteria["has_specific"] = True
            m2 = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", raw)
            if m2:
                yyyy, mm = m2.groups()
                criteria["month"] = f"{yyyy}-{mm.zfill(2)}"
                criteria["has_specific"] = True
        m_group = re.search(r"grupo\s+de\s+atribui[cç][aã]o\s*[:=]\s*([^\n\?;,.]+)", raw, flags=re.IGNORECASE)
        if m_group:
            value = m_group.group(1).strip()
            value = re.split(r"\s+(?:no|na|em|com)\s+", value, flags=re.IGNORECASE)[0].strip()
            if value:
                criteria["grupo_atribuicao"] = value
                criteria["has_specific"] = True
        return criteria

    def _wants_entity(self, q_norm: str, entity: str) -> bool:
        if entity == "chg":
            return bool(re.search(r"\bchg\b|\bchange\b|\bchanges\b|\bmudanca\b|\bmudancas\b", q_norm))
        if entity == "inc":
            return bool(re.search(r"\binc\b|\bincidente\b|\bincidentes\b", q_norm))
        return False

    def _extract_transaction_codes(self, text_value: str) -> list[str]:
        codes: list[str] = []
        # Padrão prioritário do arquivo: TRANSACAO: ZXXXX
        for m in re.finditer(r"\bTRANSACAO\s*:\s*([A-Z][A-Z0-9]{2,})\b", text_value, flags=re.IGNORECASE):
            codes.append(m.group(1).upper())
        if codes:
            return codes
        # Fallback para arquivos que não têm o marcador explícito.
        for m in re.finditer(r"\bZ[A-Z0-9]{2,}\b", text_value, flags=re.IGNORECASE):
            codes.append(m.group(0).upper())
        return codes

    def _sum_numeric_column_from_documents(self, docs: list[dict[str, Any]], question: str) -> float | None:
        import pandas as pd
        q_norm = self._norm(question)
        preferred_col = None
        m = re.search(r"(?:coluna|campo)\s+([a-zA-Z0-9_ \-]+)", question, flags=re.IGNORECASE)
        if m:
            preferred_col = self._norm(m.group(1))
        for doc in docs:
            path = doc.get("path")
            if not path:
                continue
            try:
                suffix = str(path).lower().rsplit(".", 1)[-1]
                if suffix == "csv":
                    df = pd.read_csv(path, dtype=str, keep_default_na=False)
                elif suffix in {"xlsx", "xls", "xlsm"}:
                    df = pd.read_excel(path, dtype=str).fillna("")
                else:
                    continue
                candidates = list(df.columns)
                if preferred_col:
                    candidates = [c for c in candidates if preferred_col in self._norm(c)] + candidates
                else:
                    candidates = [c for c in candidates if any(t in self._norm(c) for t in ["valor", "total", "preco", "preço"])] + candidates
                for col in candidates:
                    series = pd.to_numeric(df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
                    if series.notna().any():
                        return float(series.fillna(0).sum())
            except Exception:
                continue
        return None

    def _count_rows_from_documents(self, docs: list[dict[str, Any]]) -> int | None:
        total = 0
        found = False
        for doc in docs:
            parsed = doc.get("parsed") or {}
            meta = parsed.get("metadata") or {}
            if isinstance(meta.get("rows"), int):
                total += int(meta["rows"])
                found = True
                continue
            tables = parsed.get("tables") or []
            for t in tables:
                if isinstance(t.get("row_count"), int):
                    total += int(t["row_count"])
                    found = True
        return total if found else None

    def _direct_answer(self, case_id: str, answer: str, route: str, query_type: str, technical: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": route,
            "query_type": query_type,
            "answer_text": answer,
            "summary": answer,
            "evidence_files": [],
            "technical": {"case_id": case_id, **(technical or {})},
            "sources": {"deterministic": True},
        }

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _ensure_case_index(self, case_id: str, documents: list[dict[str, Any]]) -> None:
        if hasattr(self.retrieval_service, "ensure_case_index"):
            self.retrieval_service.ensure_case_index(case_id, documents)
            return
        if documents:
            self.retrieval_service.build_case_index(case_id, documents)

    def _build_history(self, chat_history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        history: list[dict[str, str]] = []
        for item in chat_history or []:
            if item.get("question"):
                history.append({"role": "user", "content": str(item["question"])})
            if item.get("answer_text"):
                history.append({"role": "assistant", "content": str(item["answer_text"])})
        return history[-10:]

    def _detect_focus(self, question: str, documents: list[dict[str, Any]]) -> str:
        q = self._norm(question)
        tokens = set(q.split())
        has_docs = bool(documents)
        if has_docs and tokens & self.FILE_FOCUS_TERMS:
            return "case_upload_first"
        if tokens & self.KNOWLEDGE_TERMS and not (tokens & self.FILE_FOCUS_TERMS):
            return "knowledge_base_first"
        return "hybrid_case_first" if has_docs else "knowledge_base_first"

    def _question_with_recent_context(self, question: str, history: list[dict[str, str]]) -> str:
        previous_user = [h["content"] for h in history if h.get("role") == "user"][-5:]
        return "\n".join(previous_user + [question])

    def _direct_file_evidences(self, question: str, documents: list[dict[str, Any]], focus: str) -> list[dict[str, Any]]:
        if not documents:
            return []
        q_norm = self._norm(question)
        if focus != "case_upload_first" and not any(term in q_norm for term in self.FILE_FOCUS_TERMS):
            return []
        candidates = []
        for doc in documents:
            text = ((doc.get("parsed") or {}).get("text") or "").strip()
            if not text:
                continue
            priority = 2 if self._document_source_type(doc) == "case_upload" else 1
            candidates.append((priority, doc, text))
        candidates.sort(key=lambda item: item[0], reverse=True)
        evidences = []
        for _, doc, text in candidates[:5]:
            filename = str(doc.get("filename") or "arquivo")
            evidences.append({
                "filename": filename,
                "chunk_id": f"direct_file_{doc.get('id', filename)}",
                "type": "direct_file_context",
                "score": 1.0,
                "excerpt": text[: self.max_direct_file_chars],
                "metadata": {
                    "source": "case_upload_direct",
                    "source_type": self._document_source_type(doc),
                    "document_id": doc.get("id"),
                    "content_type": doc.get("content_type"),
                },
            })
        return evidences

    def _split_text_into_records(self, text_value: str) -> list[str]:
        patterns = [
            r"(?=\n?---\s*\nsource:\s*gabbi_knowledge_article)",
            r"(?=\n?#\s*Artigo de Conhecimento Gabbi)",
            r"(?=\n?ID:\s*\d+\.\d+)",
            r"(?=\bN[úu]mero\s*:\s*(?:CHG|INC)\d{5,})",
        ]
        for pat in patterns:
            parts = [p for p in re.split(pat, text_value, flags=re.IGNORECASE) if p.strip()]
            if len(parts) > 1:
                return parts
        lines = [ln for ln in text_value.splitlines() if ln.strip()]
        if len(lines) > 5 and sum(1 for ln in lines if re.search(r"\b(CHG|INC)\d{5,}|TRANSACAO\s*:", ln, flags=re.IGNORECASE)) >= 2:
            return lines
        return [text_value]

    def _extract_tabular_evidences(self, tabular_result: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not tabular_result:
            return []
        technical = tabular_result.get("technical") or {}
        execution = technical.get("execution") or {}
        if not execution:
            return []
        table = execution.get("table") or {}
        return [{
            "filename": table.get("filename") or table.get("source") or "tabular",
            "chunk_id": "tabular_execution",
            "type": "tabular",
            "score": 1.0,
            "excerpt": json.dumps(execution, ensure_ascii=False, default=str)[:8000],
            "metadata": {"source": "tabular", "source_type": table.get("source_type")},
        }]

    def _merge_evidences(self, file_evidences, case_evidences, tabular_evidences, focus: str) -> list[dict[str, Any]]:
        merged = (file_evidences or []) + (case_evidences or []) + (tabular_evidences or [])
        if focus == "knowledge_base_first":
            merged = (tabular_evidences or []) + (file_evidences or []) + (case_evidences or [])
        out = []
        seen = set()
        for e in merged:
            key = (str(e.get("filename")), str(e.get("chunk_id")), str(e.get("type")))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out[:40]

    def _tag_evidences(self, evidences: list[dict[str, Any]], default_source: str) -> list[dict[str, Any]]:
        out = []
        for item in evidences or []:
            cloned = dict(item)
            metadata = dict(cloned.get("metadata") or {})
            metadata.setdefault("source", default_source)
            cloned["metadata"] = metadata
            out.append(cloned)
        return out

    def _ask_openai(self, question, analysis, evidences, history, mode, tabular_attempt=None, focus="hybrid_case_first") -> str | None:
        evidence_blob = "\n\n".join([
            f"[{e.get('filename')} | tipo={e.get('type')} | score={e.get('score')} | fonte={(e.get('metadata') or {}).get('source')} ]\n{e.get('excerpt')}"
            for e in evidences
        ])[:36000]
        tabular_blob = ""
        if tabular_attempt:
            tabular_blob = "\n\nResultado tabular auxiliar:\n" + json.dumps(tabular_attempt.get("technical", tabular_attempt), ensure_ascii=False, indent=2, default=str)[:9000]
        system_prompt = """
Você é um analista sênior do GABBI. Responda em português do Brasil.

REGRAS OBRIGATÓRIAS:
1. Use SOMENTE as evidências fornecidas abaixo.
2. Nunca invente números, IDs, registros, filtros ou conclusões.
3. Nunca diga que não tem acesso a arquivo quando houver evidência do tipo direct_file_context ou case_upload.
4. Para contagens/listas, respeite apenas resultados determinísticos já calculados; não recalcule por amostra.
5. Para perguntas sobre arquivo/anexo/documento, priorize o conteúdo do upload do case.
6. Se a informação não estiver nas evidências, diga exatamente que não foi encontrada nas evidências disponíveis.
""".strip()
        user_prompt = f"""
Foco detectado: {focus}

Pergunta do usuário:
{question}

Contexto analítico do caso:
{json.dumps(analysis, ensure_ascii=False, indent=2, default=str)[:8000]}

Evidências recuperadas/injetadas pelo Nexus:
{evidence_blob}
{tabular_blob}

Gere a resposta em markdown, sem extrapolar as evidências.
"""
        return self.llm_service.generate_chat(system_prompt, history[-8:], user_prompt, temperature=0)

    def _fallback_evidence_answer(self, question: str, evidences: list[dict[str, Any]]) -> str:
        lines = ["## Evidências encontradas", ""]
        for e in evidences[:10]:
            lines.append(f"### {e.get('filename')}")
            lines.append(str(e.get("excerpt", ""))[:2000])
            lines.append("")
        return "\n".join(lines)

    def _document_source_type(self, doc: dict[str, Any]) -> str:
        content_type = str(doc.get("content_type") or "").lower()
        source = str(doc.get("source") or "").lower()
        filename = str(doc.get("filename") or "").lower()
        if "database-record" in content_type or "postgres" in source or "article" in source:
            return "knowledge_base"
        if filename.startswith("gabbi_article_") or filename.startswith("gabbi_knowledge_"):
            return "knowledge_base"
        return "case_upload"

    @staticmethod
    def _norm(value: Any) -> str:
        text_value = "" if value is None else str(value)
        text_value = unicodedata.normalize("NFKD", text_value)
        text_value = "".join(ch for ch in text_value if not unicodedata.combining(ch))
        text_value = text_value.lower().strip()
        text_value = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text_value)
        return " ".join(text_value.split())
