from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter, OrderedDict
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService

try:
    from sqlalchemy import create_engine, text
    HAS_SQLALCHEMY = True
except Exception:
    HAS_SQLALCHEMY = False


class AnalysisGraphService:
    """
    Orquestrador principal do Nexus.

    Objetivo desta versão:
    - Upload/anexo do case atual é sempre evidência prioritária quando a pergunta fala de arquivo.
    - Perguntas analíticas (quantas, total, listar, todos os períodos etc.) não dependem de top_k/RAG.
    - Quando houver PostgreSQL configurado, a base Gabbi é consultada diretamente em Article/Topic/Project.
    - Se não houver banco, faz full scan dos documentos do case, não contagem por chunks recuperados.
    - A LLM apenas sintetiza/explica; não inventa contagem.
    """

    FILE_TERMS = {
        "arquivo", "arquivos", "anexo", "anexos", "anexado", "anexados", "documento", "documentos",
        "planilha", "excel", "xlsx", "xls", "csv", "txt", "pdf", "docx", "pptx", "upload",
        "conteudo", "conteúdo", "material", "do que fala", "sobre o arquivo", "sobre o documento"
    }

    ANALYTIC_TERMS = {
        "quantos", "quantas", "quantidade", "qtd", "qtde", "total", "totais", "contar",
        "conte", "número", "numero", "liste", "listar", "lista", "quais", "todos", "todas",
        "agrupado", "agrupe", "por periodo", "por período", "de todos os periodos", "de todos os períodos"
    }

    LIST_TERMS = {"liste", "listar", "lista", "quais", "todos", "todas", "mostre", "exiba", "relacione"}

    def __init__(self, retrieval_service, analysis_service):
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.llm_service = LLMService()
        self.tabular_service = TabularQueryService(llm_service=self.llm_service)

        self.direct_file_chars = int(os.getenv("GABBI_DIRECT_FILE_CONTEXT_CHARS", "24000"))
        self.max_scan_records = int(os.getenv("GABBI_DETERMINISTIC_SCAN_MAX_RECORDS", "200000"))
        self.max_answer_rows = int(os.getenv("GABBI_DETERMINISTIC_ANSWER_ROWS", "120"))

        self.database_url = (
            os.getenv("GABBI_DATABASE_URL", "").strip()
            or os.getenv("GABBI_POSTGRES_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )
        self._db_engine = None
        if HAS_SQLALCHEMY and self.database_url:
            try:
                self._db_engine = create_engine(self.database_url, future=True, pool_pre_ping=True)
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
        combined_question = self._combined_question(question, history)
        is_analytic = self._is_analytic_question(combined_question)
        is_file_question = self._is_file_question(question)

        # 1) Perguntas analíticas: tentar fonte tabular primeiro, depois banco oficial, depois full scan.
        # Isso impede o LLM de contar por top_k.
        if is_analytic:
            tabular = self._try_tabular_exact(case_id, question, documents, mode, focus)
            if tabular and not tabular.get("fallback_to_rag"):
                tabular["route"] = tabular.get("route") or "tabular_exact"
                tabular["query_type"] = tabular.get("query_type") or "structured"
                return tabular

            db_answer = self._deterministic_from_postgres(
                case_id=case_id,
                question=question,
                combined_question=combined_question,
                documents=documents,
            )
            if db_answer:
                return db_answer

            scan_answer = self._deterministic_from_documents(
                case_id=case_id,
                question=question,
                combined_question=combined_question,
                documents=documents,
            )
            if scan_answer:
                return scan_answer

        # 2) Perguntas sobre arquivo/anexo: injetar conteúdo real do upload antes do RAG.
        direct_file_evidences = self._direct_file_evidences(question, documents) if is_file_question else []

        # 3) RAG do case atual. O RetrievalService já filtra por case_id.
        top_k = 28 if (is_file_question or focus == "case_upload_first") else 16
        case_evidences = self.retrieval_service.search(case_id, question, top_k=top_k)
        case_evidences = self._tag_evidences(case_evidences, "case_current_index")

        # 4) Tabular como complemento para perguntas não analíticas.
        tabular_result = self._try_tabular_exact(case_id, question, documents, mode, focus)
        tabular_evidences = self._extract_tabular_evidences(tabular_result)

        evidences = self._merge_evidences(direct_file_evidences, case_evidences, tabular_evidences, focus)

        formatted = self.analysis_service.format_answer(question, evidences, analysis, mode=mode)

        if self.llm_service.status()["enabled"] and evidences:
            answer = self._ask_openai(
                question=question,
                analysis=analysis,
                evidences=evidences,
                history=history,
                mode=mode,
                focus=focus,
                tabular_attempt=tabular_result,
            )
            if answer:
                formatted.update(
                    {
                        "summary": answer,
                        "answer_text": answer,
                        "route": "hybrid_evidence_exact_guard",
                        "query_type": "document_qa",
                        "sources": {
                            "focus": focus,
                            "case_id": case_id,
                            "documents_in_case": len(documents or []),
                            "direct_file_evidence": bool(direct_file_evidences),
                            "rag_evidences": len(case_evidences),
                            "tabular": bool(tabular_result and not tabular_result.get("fallback_to_rag")),
                            "conversation_memory": bool(history),
                        },
                    }
                )
                return formatted

        if evidences:
            answer = self._fallback_evidence_answer(evidences)
            formatted.update(
                {
                    "summary": answer,
                    "answer_text": answer,
                    "route": "evidence_fallback",
                    "query_type": "document_qa",
                }
            )
            return formatted

        msg = (
            "Não encontrei evidência textual suficiente nos documentos do case atual nem na base configurada "
            "para responder com precisão."
        )
        formatted.update({"summary": msg, "answer_text": msg, "route": "no_evidence", "query_type": "no_evidence"})
        return formatted

    # ------------------------------------------------------------------
    # Indexação e foco
    # ------------------------------------------------------------------

    def _ensure_case_index(self, case_id: str, documents: list[dict[str, Any]]) -> None:
        if hasattr(self.retrieval_service, "ensure_case_index"):
            self.retrieval_service.ensure_case_index(case_id, documents)
        elif documents:
            self.retrieval_service.build_case_index(case_id, documents)

    def _build_history(self, chat_history: list[dict[str, Any]] | None) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for item in chat_history or []:
            if item.get("question"):
                out.append({"role": "user", "content": str(item.get("question"))})
            if item.get("answer_text"):
                out.append({"role": "assistant", "content": str(item.get("answer_text"))})
        return out[-10:]

    def _combined_question(self, question: str, history: list[dict[str, str]]) -> str:
        previous_users = [h["content"] for h in history if h.get("role") == "user"][-3:]
        return "\n".join(previous_users + [question])

    def _detect_focus(self, question: str, documents: list[dict[str, Any]]) -> str:
        q = self._norm(question)
        if documents and self._is_file_question(q):
            return "case_upload_first"
        if any(t in q for t in ["base", "conhecimento", "treinamento", "artigo", "artigos", "banco", "postgres"]):
            return "knowledge_base_first"
        return "hybrid_case_first" if documents else "knowledge_base_first"

    def _is_file_question(self, question: str) -> bool:
        q = self._norm(question)
        return any(term in q for term in self.FILE_TERMS)

    def _is_analytic_question(self, question: str) -> bool:
        q = self._norm(question)
        return any(term in q for term in self.ANALYTIC_TERMS)

    # ------------------------------------------------------------------
    # Tabular
    # ------------------------------------------------------------------

    def _try_tabular_exact(self, case_id: str, question: str, documents: list[dict[str, Any]], mode: str, focus: str):
        try:
            return self.tabular_service.answer_question(
                case_id=case_id,
                question=question,
                documents=documents,
                mode=mode,
                source_preference=focus,
            )
        except TypeError:
            try:
                return self.tabular_service.answer_question(case_id, question, documents, mode=mode)
            except Exception:
                return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Determinístico via PostgreSQL oficial Article/Topic/Project
    # ------------------------------------------------------------------

    def _deterministic_from_postgres(
        self,
        *,
        case_id: str,
        question: str,
        combined_question: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if self._db_engine is None:
            return None

        criteria = self._extract_criteria(combined_question)
        if not criteria.get("has_any"):
            return None

        project_ids = self._extract_project_ids_from_documents(documents)
        rows = self._fetch_postgres_articles(project_ids=project_ids)
        if not rows:
            return None

        records = []
        for row in rows:
            text_blob = "\n".join(
                [
                    f"Projeto: {row.get('project_name') or ''}",
                    f"project_id: {row.get('project_id') or ''}",
                    f"Tópico: {row.get('topic_name') or ''}",
                    f"Descrição do tópico: {row.get('topic_description') or ''}",
                    f"Referência do artigo: {row.get('article_ref_id') or ''}",
                    str(row.get("article_text") or ""),
                ]
            )
            records.append(
                {
                    "filename": "PostgreSQL:public.Article",
                    "record_id": str(row.get("article_id") or row.get("article_ref_id") or ""),
                    "text": text_blob,
                    "source_type": "postgres_article",
                    "raw": row,
                }
            )

        matched = [r for r in records if self._record_matches(r, criteria)]
        if not matched:
            return self._no_match_if_specific(case_id, question, criteria, len(records), "PostgreSQL:public.Article")

        return self._format_deterministic_result(
            case_id=case_id,
            question=question,
            criteria=criteria,
            records_scanned=len(records),
            matched=matched,
            source="PostgreSQL:public.Article",
        )

    def _fetch_postgres_articles(self, project_ids: list[str]) -> list[dict[str, Any]]:
        where = [
            "coalesce(A.deleted, false) = false",
            "coalesce(A.published, false) = true",
            "coalesce(T.deleted, false) = false",
            "coalesce(T.active, true) = true",
            "coalesce(P.deleted, false) = false",
            "length(trim(coalesce(A.article, ''))) > 0",
        ]
        params: dict[str, Any] = {}
        if project_ids:
            where.append('lower(T."projectId") = ANY(:project_ids)')
            params["project_ids"] = [p.lower() for p in project_ids]

        sql = text(
            f"""
            SELECT
                A.id AS article_id,
                A."refId" AS article_ref_id,
                A.article AS article_text,
                A."createdOn" AS article_created_on,
                A."updatedOn" AS article_updated_on,
                T.id AS topic_id,
                T."refId" AS topic_ref_id,
                T.name AS topic_name,
                T.description AS topic_description,
                T."projectId" AS project_id,
                P.name AS project_name
            FROM public."Article" A
            INNER JOIN public."Topic" T ON T.id = A."topicId"
            INNER JOIN public."Project" P ON P.id = T."projectId"
            WHERE {" AND ".join(where)}
            ORDER BY T."refId", A."refId"
            """
        )
        try:
            with self._db_engine.connect() as conn:
                result = conn.execute(sql, params)
                return [dict(row._mapping) for row in result]
        except Exception:
            return []

    def _extract_project_ids_from_documents(self, documents: list[dict[str, Any]]) -> list[str]:
        found: list[str] = []
        seen = set()
        for doc in documents or []:
            candidates = []
            md = doc.get("metadata") or {}
            for k in ("project_id", "projectId", "project"):
                if md.get(k):
                    candidates.append(str(md.get(k)))
            text = ((doc.get("parsed") or {}).get("text") or "")
            candidates.extend(re.findall(r"(?im)^project_id:\s*([0-9a-fA-F\-]{8,}|[A-Za-z0-9_\-]+)\s*$", text))
            for c in candidates:
                v = str(c).strip().lower()
                if v and v not in seen:
                    seen.add(v)
                    found.append(v)
        return found

    # ------------------------------------------------------------------
    # Determinístico em documentos do case
    # ------------------------------------------------------------------

    def _deterministic_from_documents(
        self,
        *,
        case_id: str,
        question: str,
        combined_question: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        criteria = self._extract_criteria(combined_question)
        if not criteria.get("has_any"):
            return None

        records = self._extract_records_from_documents(documents)
        if not records:
            return None

        matched = [r for r in records[: self.max_scan_records] if self._record_matches(r, criteria)]
        if not matched:
            return self._no_match_if_specific(case_id, question, criteria, len(records), "documentos_do_case")

        return self._format_deterministic_result(
            case_id=case_id,
            question=question,
            criteria=criteria,
            records_scanned=len(records),
            matched=matched,
            source="documentos_do_case",
        )

    def _extract_records_from_documents(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for doc in documents or []:
            text_value = ((doc.get("parsed") or {}).get("text") or "").strip()
            if not text_value:
                continue
            filename = str(doc.get("filename") or "arquivo")
            source_type = self._document_source_type(doc)
            for i, part in enumerate(self._split_records(text_value), start=1):
                if len(part.strip()) < 20:
                    continue
                records.append(
                    {
                        "filename": filename,
                        "record_id": f"{doc.get('id', filename)}_{i}",
                        "text": part.strip(),
                        "source_type": source_type,
                    }
                )
        return records

    def _split_records(self, text_value: str) -> list[str]:
        # Bundles de conhecimento Gabbi.
        for pat in [
            r"(?=\n?---\s*\nsource:\s*gabbi_knowledge_article)",
            r"(?=\n?#\s*Artigo de Conhecimento Gabbi)",
        ]:
            parts = [p for p in re.split(pat, text_value, flags=re.IGNORECASE) if p.strip()]
            if len(parts) > 1:
                return parts

        # CSV/textos tabulares parseados: divide por linhas quando houver muitos códigos.
        lines = [ln for ln in text_value.splitlines() if ln.strip()]
        code_lines = [ln for ln in lines if re.search(r"\b(CHG|INC|REQ|RITM|TASK)\d{4,}\b", ln, flags=re.I)]
        if len(code_lines) >= 2:
            return lines

        return [text_value]

    # ------------------------------------------------------------------
    # Critérios, matching e formatação determinística
    # ------------------------------------------------------------------

    def _extract_criteria(self, text_value: str) -> dict[str, Any]:
        raw = text_value or ""
        norm = self._norm(raw)
        criteria: dict[str, Any] = {
            "has_any": False,
            "code_type": None,
            "topics": [],
            "months": [],
            "field_equals": {},
            "codes": [],
            "label_parts": [],
            "list_mode": bool(set(norm.split()) & self.LIST_TERMS),
            "all_periods": any(x in norm for x in ["todos os periodos", "todos os períodos", "todos periodos", "todos períodos", "de todos"]),
        }

        # Tipo de entidade.
        if re.search(r"\b(chg|change|changes|mudanca|mudança|mudancas|mudanças)\b", norm, flags=re.I):
            criteria["code_type"] = "CHG"
            criteria["label_parts"].append("tipo=CHG")
        elif re.search(r"\b(inc|incidente|incidentes|chamado|chamados)\b", norm, flags=re.I):
            criteria["code_type"] = "INC"
            criteria["label_parts"].append("tipo=INC")

        # Tópico tipo CHG:10-2025.
        for topic in re.findall(r"\b(CHG|INC)\s*:\s*(0?[1-9]|1[0-2])[-/](20\d{2})\b", raw, flags=re.I):
            prefix, mm, yyyy = topic
            t = f"{prefix.upper()}:{mm.zfill(2)}-{yyyy}"
            criteria["topics"].append(t)
            criteria["months"].append(f"{yyyy}-{mm.zfill(2)}")
            criteria["code_type"] = prefix.upper()
            criteria["label_parts"].append(t)

        # Mês explícito.
        for mm, yyyy in re.findall(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", raw):
            criteria["months"].append(f"{yyyy}-{mm.zfill(2)}")
            criteria["label_parts"].append(f"mes={yyyy}-{mm.zfill(2)}")
        for yyyy, mm in re.findall(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", raw):
            criteria["months"].append(f"{yyyy}-{mm.zfill(2)}")
            criteria["label_parts"].append(f"mes={yyyy}-{mm.zfill(2)}")

        # Grupo de atribuição: valor.
        m = re.search(r"grupo\s+de\s+atribui[cç][aã]o\s*[:=]\s*([A-Za-z0-9_\-\. ]+)", raw, flags=re.I)
        if m:
            value = m.group(1).strip()
            value = re.split(r"\s+(?:no|na|em|com|e)\s+", value, flags=re.I)[0].strip(" ?.,;")
            if value:
                criteria["field_equals"]["grupo_atribuicao"] = value
                criteria["label_parts"].append(f"grupo_atribuicao={value}")

        # Códigos específicos.
        for code in re.findall(r"\b(CHG\d{5,}|INC\d{5,}|REQ\d{4,}|RITM\d{4,}|TASK\d{4,})\b", raw, flags=re.I):
            c = code.upper()
            criteria["codes"].append(c)
            criteria["label_parts"].append(c)

        # Dedup.
        for key in ("topics", "months", "codes"):
            dedup = []
            seen = set()
            for item in criteria[key]:
                if item not in seen:
                    seen.add(item)
                    dedup.append(item)
            criteria[key] = dedup

        criteria["has_any"] = bool(
            criteria.get("code_type")
            or criteria.get("topics")
            or criteria.get("months")
            or criteria.get("field_equals")
            or criteria.get("codes")
        )
        criteria["label"] = ", ".join(OrderedDict.fromkeys(criteria["label_parts"]).keys())
        return criteria

    def _record_matches(self, record: dict[str, Any], criteria: dict[str, Any]) -> bool:
        text_value = record.get("text", "") or ""
        text_norm = self._norm(text_value)

        code_type = criteria.get("code_type")
        if code_type and not re.search(rf"\b{code_type}\d{{5,}}\b", text_value, flags=re.I):
            return False

        for code in criteria.get("codes") or []:
            if self._norm(code) not in text_norm:
                return False

        for topic in criteria.get("topics") or []:
            prefix, rest = topic.split(":", 1)
            mm, yyyy = rest.split("-")
            variants = [topic, f"{prefix}:{mm}/{yyyy}", f"{yyyy}-{mm}", f"{mm}-{yyyy}", f"Mês: {yyyy}-{mm}", f"Mes: {yyyy}-{mm}"]
            if not any(self._norm(v) in text_norm for v in variants):
                return False

        for month in criteria.get("months") or []:
            yyyy, mm = month.split("-")
            variants = [month, f"{mm}-{yyyy}", f"{mm}/{yyyy}", f"Mês: {month}", f"Mes: {month}"]
            if not any(self._norm(v) in text_norm for v in variants):
                return False

        field_equals = criteria.get("field_equals") or {}
        if field_equals.get("grupo_atribuicao"):
            expected = self._norm(field_equals["grupo_atribuicao"])
            actual = self._extract_field(text_value, "grupo_atribuicao")
            if actual:
                if self._norm(actual) != expected:
                    return False
            elif expected not in text_norm:
                return False

        return True

    def _extract_field(self, text_value: str, field: str) -> str:
        if field == "grupo_atribuicao":
            patterns = [
                r"Grupo de atribui[cç][aã]o\s*[:\-]\s*(.+?)(?=\s+(?:Tipo de Indisponibilidade|Tipo|Estado|Status|Data|IC Impactado|Solicita[cç][aã]o|Aberta por|Atribu[ií]do a)\s*[:\-]|$)",
                r"grupo_atribuicao[\"']?\s*[:=]\s*[\"']?([^\"',}\n\r]+)",
            ]
        else:
            patterns = []
        for pat in patterns:
            m = re.search(pat, text_value, flags=re.I | re.S)
            if m:
                return re.sub(r"\s+", " ", m.group(1).strip()).strip(" ,.;")
        return ""

    def _format_deterministic_result(
        self,
        *,
        case_id: str,
        question: str,
        criteria: dict[str, Any],
        records_scanned: int,
        matched: list[dict[str, Any]],
        source: str,
    ) -> dict[str, Any]:
        items = OrderedDict()
        for r in matched:
            code = self._main_code(r.get("text", "")) or r.get("record_id") or str(len(items) + 1)
            if code not in items:
                items[code] = r

        rows = [self._summarize_record(code, r.get("text", "")) for code, r in items.items()]
        period_counter = Counter()
        for r in matched:
            month = self._extract_month(r.get("text", ""))
            if month:
                period_counter[month] += 1

        is_list = criteria.get("list_mode")
        title = "## Resposta direta"
        if is_list:
            first_line = f"Foram encontrados **{len(items)}** registro(s) distintos correspondentes aos critérios."
        else:
            first_line = f"Foram encontrados **{len(items)}** registro(s) distintos correspondentes aos critérios."

        lines = [
            title,
            "",
            first_line,
            "",
            "## Critérios aplicados",
        ]
        if criteria.get("label"):
            for part in criteria["label"].split(", "):
                lines.append(f"- `{part}`")
        else:
            lines.append("- Critério analítico identificado na pergunta.")

        lines.extend(["", f"Fonte determinística: **{source}**.", f"Registros avaliados: **{records_scanned}**."])

        if period_counter and criteria.get("all_periods"):
            lines.extend(["", "## Distribuição por período"])
            for month, total in sorted(period_counter.items()):
                lines.append(f"- **{month}**: {total}")

        if rows:
            lines.extend(["", "## Registros encontrados"])
            for idx, row in enumerate(rows[: self.max_answer_rows], start=1):
                detail = " – ".join([v for v in [row.get("tipo"), row.get("estado"), row.get("grupo_atribuicao")] if v])
                lines.append(f"{idx}. **{row.get('numero')}**" + (f" – {detail}" if detail else ""))
            if len(rows) > self.max_answer_rows:
                lines.append(f"\n> Lista truncada na resposta: exibindo {self.max_answer_rows} de {len(rows)} registros.")

        lines.append("\n> Resultado gerado por varredura determinística, não por aproximação semântica/top-k.")

        answer = "\n".join(lines)
        return {
            "route": "deterministic_exact",
            "query_type": "list" if is_list else "count",
            "answer_text": answer,
            "summary": answer,
            "evidence_files": sorted({r.get("filename", "") for r in matched if r.get("filename")}),
            "technical": {
                "case_id": case_id,
                "source": source,
                "records_scanned": records_scanned,
                "records_matched": len(matched),
                "unique_items": len(items),
                "criteria": criteria,
            },
            "sources": {"deterministic": True, "source": source},
        }

    def _no_match_if_specific(self, case_id: str, question: str, criteria: dict[str, Any], scanned: int, source: str) -> dict[str, Any] | None:
        # Se há critério explícito, retornar zero é melhor que deixar LLM tentar adivinhar.
        if criteria.get("has_any"):
            answer = (
                "## Resposta direta\n\n"
                "Foram encontrados **0** registro(s) correspondentes aos critérios informados.\n\n"
                "## Critérios aplicados\n"
                + "\n".join([f"- `{p}`" for p in (criteria.get("label") or "critério informado").split(", ")])
                + f"\n\nFonte determinística: **{source}**.\nRegistros avaliados: **{scanned}**."
            )
            return {
                "route": "deterministic_exact",
                "query_type": "count",
                "answer_text": answer,
                "summary": answer,
                "technical": {"case_id": case_id, "source": source, "records_scanned": scanned, "records_matched": 0, "criteria": criteria},
            }
        return None

    def _main_code(self, text_value: str) -> str | None:
        m = re.search(r"\b(CHG\d{5,}|INC\d{5,}|REQ\d{4,}|RITM\d{4,}|TASK\d{4,})\b", text_value, flags=re.I)
        return m.group(1).upper() if m else None

    def _extract_month(self, text_value: str) -> str:
        for pat in [r"\bM[eê]s\s*[:\-]\s*(20\d{2}[-/](0?[1-9]|1[0-2]))", r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b"]:
            m = re.search(pat, text_value, flags=re.I)
            if m:
                if "-" in m.group(1) or "/" in m.group(1):
                    yyyy, mm = re.split(r"[-/]", m.group(1))
                    return f"{yyyy}-{mm.zfill(2)}"
                return f"{m.group(1)}-{m.group(2).zfill(2)}"
        m = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", text_value)
        if m:
            return f"{m.group(2)}-{m.group(1).zfill(2)}"
        return ""

    def _summarize_record(self, code: str, text_value: str) -> dict[str, str]:
        def first(patterns: list[str]) -> str:
            for pat in patterns:
                m = re.search(pat, text_value, flags=re.I | re.S)
                if m:
                    return re.sub(r"\s+", " ", m.group(1).strip()).strip(" ,.;")[:180]
            return ""

        return {
            "numero": code,
            "tipo": first([r"\bTipo\s*[:\-]\s*(.+?)(?=\s+(?:Estado|Status|Data|IC Impactado|Grupo de atribui|Solicita|Aberta por)\s*[:\-]|$)"]),
            "estado": first([r"\bEstado\s*[:\-]\s*(.+?)(?=\s+(?:Status|Data|IC Impactado|Grupo de atribui|Tipo|Solicita|Aberta por)\s*[:\-]|$)"]),
            "grupo_atribuicao": first([r"\bGrupo de atribui[cç][aã]o\s*[:\-]\s*(.+?)(?=\s+(?:Tipo de Indisponibilidade|Tipo|Estado|Status|Data|IC Impactado|Solicita|Aberta por|Atribu[ií]do a)\s*[:\-]|$)"]),
        }

    # ------------------------------------------------------------------
    # Arquivos e evidências
    # ------------------------------------------------------------------

    def _direct_file_evidences(self, question: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not documents:
            return []
        requested = self._requested_filenames(question)
        candidates = []
        for doc in documents:
            text_value = ((doc.get("parsed") or {}).get("text") or "").strip()
            if not text_value:
                continue
            filename = str(doc.get("filename") or "arquivo")
            source_type = self._document_source_type(doc)
            if source_type != "case_upload":
                continue
            if requested:
                f_norm = self._norm(filename)
                if not any(req in f_norm or f_norm in req for req in requested):
                    continue
            candidates.append((filename, doc, text_value))

        if not candidates:
            # Se não houver upload classificado, ainda usa qualquer documento textual do case para não dizer que não achou.
            for doc in documents:
                text_value = ((doc.get("parsed") or {}).get("text") or "").strip()
                if text_value:
                    candidates.append((str(doc.get("filename") or "arquivo"), doc, text_value))

        evidences = []
        for filename, doc, text_value in candidates[:3]:
            evidences.append(
                {
                    "filename": filename,
                    "chunk_id": f"direct_file_{doc.get('id', filename)}",
                    "type": "direct_file_context",
                    "score": 1.0,
                    "excerpt": text_value[: self.direct_file_chars],
                    "metadata": {
                        "source": "case_upload_direct",
                        "source_type": self._document_source_type(doc),
                        "document_id": doc.get("id"),
                        "content_type": doc.get("content_type"),
                    },
                }
            )
        return evidences

    def _requested_filenames(self, question: str) -> list[str]:
        return [self._norm(x.strip()) for x in re.findall(r"([\w\-. ]+\.(?:txt|csv|xlsx|xls|pdf|docx|pptx|md|markdown))", question, flags=re.I)]

    def _extract_tabular_evidences(self, tabular_result: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not tabular_result:
            return []
        technical = tabular_result.get("technical") or {}
        execution = technical.get("execution") or {}
        if not execution:
            return []
        table = execution.get("table") or {}
        return [
            {
                "filename": table.get("source") or table.get("filename") or "tabular",
                "chunk_id": "tabular_execution",
                "type": "tabular_execution",
                "score": 1.0,
                "excerpt": json.dumps(execution, ensure_ascii=False, default=str)[:8000],
                "metadata": {"source": "tabular", "source_type": table.get("source_type")},
            }
        ]

    def _merge_evidences(self, direct_files, case_evidences, tabular_evidences, focus: str) -> list[dict[str, Any]]:
        if focus in {"case_upload_first", "hybrid_case_first"}:
            merged = (direct_files or []) + (case_evidences or []) + (tabular_evidences or [])
        else:
            merged = (tabular_evidences or []) + (direct_files or []) + (case_evidences or [])
        out = []
        seen = set()
        for e in merged:
            key = (str(e.get("filename")), str(e.get("chunk_id")), str(e.get("type")))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out[:36]

    def _tag_evidences(self, evidences: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
        out = []
        for e in evidences or []:
            cloned = dict(e)
            md = dict(cloned.get("metadata") or {})
            md.setdefault("source", source)
            cloned["metadata"] = md
            out.append(cloned)
        return out

    def _ask_openai(self, question, analysis, evidences, history, mode, focus, tabular_attempt=None) -> str | None:
        evidence_blob = "\n\n".join(
            [
                f"[{e.get('filename')} | tipo={e.get('type')} | score={e.get('score')} | fonte={(e.get('metadata') or {}).get('source')}]\n{e.get('excerpt')}"
                for e in evidences
            ]
        )[:30000]
        tabular_blob = ""
        if tabular_attempt:
            tabular_blob = "\n\nResultado tabular auxiliar:\n" + json.dumps(tabular_attempt.get("technical", tabular_attempt), ensure_ascii=False, indent=2, default=str)[:8000]

        system_prompt = """
Você é um analista sênior do GABBI. Responda em português do Brasil.

REGRAS OBRIGATÓRIAS:
1. Use somente as evidências fornecidas.
2. Se a evidência existir, responda exatamente o que está nela.
3. Nunca diga que não tem acesso ao arquivo quando houver evidência direct_file_context ou case_upload_direct.
4. Para números, totais, contagens e listas, nunca estime; use somente resultados determinísticos/tabulares já executados.
5. Se a informação não estiver nas evidências, diga que não foi encontrada nas evidências disponíveis.
6. Não invente IDs, registros, filtros, números ou conclusões.
7. Para perguntas sobre arquivo/anexo, priorize o arquivo do case atual.
8. A base Gabbi/PostgreSQL complementa o arquivo, mas não substitui upload do usuário.
""".strip()
        if mode == "executive":
            system_prompt += "\nUse linguagem executiva, objetiva e direta."
        elif mode == "technical":
            system_prompt += "\nUse linguagem técnica e precisa."

        user_prompt = f"""
Foco detectado: {focus}

Pergunta:
{question}

Contexto analítico:
{json.dumps(analysis, ensure_ascii=False, indent=2, default=str)[:8000]}

Evidências:
{evidence_blob}
{tabular_blob}

Gere resposta em markdown, sem extrapolar.
"""
        return self.llm_service.generate_chat(system_prompt, history[-8:], user_prompt, temperature=0)

    def _fallback_evidence_answer(self, evidences: list[dict[str, Any]]) -> str:
        lines = ["## Evidências encontradas", ""]
        for e in evidences[:10]:
            lines.append(f"### {e.get('filename')}")
            lines.append(str(e.get("excerpt", ""))[:1600])
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
