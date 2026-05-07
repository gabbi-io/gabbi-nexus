from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any

from app.services.llm import LLMService
from app.services.tabular import TabularQueryService


class AnalysisGraphService:
    """Orquestrador do chat do Nexus.

    Regras implementadas nesta versão:
    - Nunca responde por aproximação em perguntas quantitativas quando há texto disponível.
    - Sempre reindexa/garante o índice do case atual antes de responder.
    - Para perguntas sobre arquivo/anexo, injeta o conteúdo real do upload como evidência prioritária.
    - Para perguntas de contagem/listagem, faz varredura determinística em TODOS os documentos do case.
    - A LLM só sintetiza quando não há resposta determinística direta.
    """

    FILE_FOCUS_TERMS = {
        "arquivo", "arquivos", "anexo", "anexos", "anexado", "anexados", "documento", "documentos",
        "planilha", "excel", "pdf", "docx", "csv", "txt", "upload", "uploads", "enviado", "enviados",
        "teor", "escopo", "conteudo", "conteúdo", "material"
    }
    COUNT_TERMS = {"quantos", "quantas", "quantidade", "qtd", "qtde", "total", "contar", "conte", "número", "numero"}
    LIST_TERMS = {"liste", "listar", "lista", "quais", "mostre", "exiba", "relacione", "detalhe"}
    KNOWLEDGE_TERMS = {"base", "conhecimento", "treinamento", "treinado", "treinada", "artigo", "artigos", "topic", "tópico", "topico"}

    def __init__(self, retrieval_service, analysis_service):
        self.retrieval_service = retrieval_service
        self.analysis_service = analysis_service
        self.llm_service = LLMService()
        self.tabular_service = TabularQueryService(llm_service=self.llm_service)
        self.max_direct_file_chars = int(os.getenv("GABBI_DIRECT_FILE_CONTEXT_CHARS", "14000"))
        self.max_scan_records = int(os.getenv("GABBI_DETERMINISTIC_SCAN_MAX_RECORDS", "20000"))

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

        # 1) Resposta determinística para contagens/listagens. Não deixa LLM "chutar".
        deterministic = self._deterministic_answer(
            case_id=case_id,
            question=question,
            question_with_context=question_with_context,
            documents=documents,
            history=history,
        )
        if deterministic:
            return deterministic

        # 2) Evidência direta de arquivos anexados. Isso impede resposta "não tenho acesso ao arquivo".
        file_evidences = self._direct_file_evidences(question, documents, focus)

        # 3) RAG do case atual.
        top_k = 24 if focus in {"case_upload_first", "hybrid_case_first"} else 16
        case_evidences = self.retrieval_service.search(case_id, question, top_k=top_k)
        case_evidences = self._tag_evidences(case_evidences, "case_upload_or_knowledge_in_case")

        # 4) Consulta tabular só entra como complemento, nunca substitui upload.
        tabular_result = None
        try:
            tabular_result = self.tabular_service.answer_question(case_id, question, documents, mode=mode)
        except TypeError:
            tabular_result = self.tabular_service.answer_question(case_id, question, documents, mode=mode)
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

        # Fallback sem LLM: mostra evidências reais, nunca diz que não achou se existe texto.
        if evidences:
            answer = self._fallback_evidence_answer(question, evidences)
            formatted.update({"summary": answer, "answer_text": answer, "route": "evidence_fallback", "query_type": "document_qa"})
            return formatted

        formatted.update({
            "answer_text": "Não encontrei evidência textual suficiente nos documentos indexados deste case para responder com precisão.",
            "summary": "Não encontrei evidência textual suficiente nos documentos indexados deste case para responder com precisão.",
            "route": "no_evidence",
            "query_type": "document_qa",
        })
        return formatted

    def _ensure_case_index(self, case_id: str, documents: list[dict[str, Any]]) -> None:
        if hasattr(self.retrieval_service, "ensure_case_index"):
            self.retrieval_service.ensure_case_index(case_id, documents)
            return
        status = getattr(self.retrieval_service, "status", lambda: {})()
        # Reindexa sempre; é intencional para evitar índice em memória perdido após restart.
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
        previous_user = [h["content"] for h in history if h.get("role") == "user"][-3:]
        return "\n".join(previous_user + [question])

    def _direct_file_evidences(self, question: str, documents: list[dict[str, Any]], focus: str) -> list[dict[str, Any]]:
        if not documents:
            return []
        q_norm = self._norm(question)
        if focus != "case_upload_first" and not any(term in q_norm for term in self.FILE_FOCUS_TERMS):
            return []

        requested_names = self._extract_requested_filenames(question)
        candidates = []
        for doc in documents:
            filename = str(doc.get("filename") or "arquivo")
            source_type = self._document_source_type(doc)
            text = ((doc.get("parsed") or {}).get("text") or "").strip()
            if not text:
                continue
            if requested_names:
                f_norm = self._norm(filename)
                if not any(name in f_norm or f_norm in name for name in requested_names):
                    continue
            # Para pergunta de arquivo, prefira uploads reais; se só existir base, usa o que houver.
            priority = 2 if source_type == "case_upload" else 1
            candidates.append((priority, doc, text))

        if not candidates and documents:
            for doc in documents:
                text = ((doc.get("parsed") or {}).get("text") or "").strip()
                if text:
                    candidates.append((1, doc, text))

        candidates.sort(key=lambda item: item[0], reverse=True)
        evidences = []
        for _, doc, text in candidates[:3]:
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

    def _deterministic_answer(
        self,
        case_id: str,
        question: str,
        question_with_context: str,
        documents: list[dict[str, Any]],
        history: list[dict[str, str]],
    ) -> dict[str, Any] | None:
        q = self._norm(question)
        tokens = set(q.split())
        is_count = bool(tokens & self.COUNT_TERMS)
        is_list = bool(tokens & self.LIST_TERMS)
        if not (is_count or is_list):
            return None

        records = self._extract_records(documents)
        if not records:
            return None

        criteria = self._extract_criteria(question_with_context)
        if not criteria.get("has_any"):
            return None

        matched = []
        for record in records[: self.max_scan_records]:
            if self._record_matches(record, criteria):
                matched.append(record)

        # Em pergunta quantitativa, se tenho dados e critério, respondo determinístico.
        label = criteria.get("label") or "critério informado"
        unique_codes: dict[str, dict[str, Any]] = {}
        for r in matched:
            code = self._extract_main_code(r["text"]) or r.get("record_id") or str(len(unique_codes) + 1)
            unique_codes.setdefault(code, r)

        if is_count:
            count = len(unique_codes)
            rows = []
            for code, r in list(unique_codes.items())[:60]:
                rows.append(self._summarize_record(code, r["text"]))
            answer = self._format_deterministic_count_answer(question, label, count, rows, criteria, len(records))
        else:
            rows = []
            for code, r in list(unique_codes.items())[:100]:
                rows.append(self._summarize_record(code, r["text"]))
            answer = self._format_deterministic_list_answer(question, label, rows, criteria, len(records))

        return {
            "route": "deterministic_full_scan",
            "query_type": "count" if is_count else "list",
            "answer_text": answer,
            "summary": answer,
            "evidence_files": sorted({r.get("filename", "") for r in matched if r.get("filename")}),
            "technical": {
                "case_id": case_id,
                "records_scanned": len(records),
                "records_matched": len(matched),
                "unique_items": len(unique_codes),
                "criteria": criteria,
                "note": "Resposta gerada por varredura determinística de todos os documentos textuais do case, não por top_k semântico.",
            },
            "sources": {
                "documents_in_case": len(documents or []),
                "deterministic_full_scan": True,
            },
        }

    def _extract_records(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for doc in documents or []:
            filename = str(doc.get("filename") or "arquivo")
            text = ((doc.get("parsed") or {}).get("text") or "").strip()
            if not text:
                continue
            parts = self._split_text_into_records(text)
            for i, part in enumerate(parts, start=1):
                p = part.strip()
                if len(p) < 20:
                    continue
                records.append({
                    "filename": filename,
                    "record_id": f"{doc.get('id', filename)}_{i}",
                    "text": p,
                    "source_type": self._document_source_type(doc),
                })
        return records

    def _split_text_into_records(self, text: str) -> list[str]:
        # Bundles do Gabbi: cada artigo começa com front matter/source ou título.
        patterns = [
            r"(?=\n?---\s*\nsource:\s*gabbi_knowledge_article)",
            r"(?=\n?#\s*Artigo de Conhecimento Gabbi)",
            r"(?=\n?ID:\s*\d+\.\d+)",
        ]
        for pat in patterns:
            parts = [p for p in re.split(pat, text, flags=re.IGNORECASE) if p.strip()]
            if len(parts) > 1:
                return parts
        # CSV/texto colado com registros técnicos: cada linha grande pode ser um registro.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(lines) > 5 and sum(1 for ln in lines if "CHG" in ln.upper() or "INC" in ln.upper() or "Código:" in ln) >= 2:
            return lines
        return [text]

    def _extract_criteria(self, text: str) -> dict[str, Any]:
        raw = text or ""
        norm = self._norm(raw)
        criteria: dict[str, Any] = {"has_any": False, "must_contains": [], "label_parts": []}

        # Tópicos do tipo CHG:10-2025 / INC:10-2025
        for topic in re.findall(r"\b(?:CHG|INC)\s*:\s*\d{1,2}[-/]\d{4}\b", raw, flags=re.IGNORECASE):
            t = topic.upper().replace(" ", "")
            criteria["must_contains"].append(t)
            criteria["label_parts"].append(t)

        # Mês explícito 10-2025 vira 2025-10 e também mantém 10-2025.
        for mm, yyyy in re.findall(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", raw):
            criteria["must_contains"].extend([f"{mm.zfill(2)}-{yyyy}", f"{yyyy}-{mm.zfill(2)}"])
            criteria["label_parts"].append(f"{yyyy}-{mm.zfill(2)}")

        # Campo explícito: Grupo de atribuição: X
        m = re.search(r"grupo\s+de\s+atribui[cç][aã]o\s*[:=]\s*([^\n\?;,.]+)", raw, flags=re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            value = re.split(r"\s+(?:no|na|em|com)\s+", value, flags=re.IGNORECASE)[0].strip()
            if value:
                criteria["must_contains"].append(value)
                criteria["label_parts"].append(f"Grupo de atribuição={value}")

        # Códigos exatos e grupos técnicos em maiúsculo.
        for code in re.findall(r"\b(?:CHG|INC|REQ|RITM|TASK)\d{4,}\b", raw, flags=re.IGNORECASE):
            criteria["must_contains"].append(code.upper())
            criteria["label_parts"].append(code.upper())
        for group in re.findall(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+){1,}\b", raw):
            if group.upper() not in {"VIVO", "CHG"}:
                criteria["must_contains"].append(group)
                criteria["label_parts"].append(group)

        # Remove duplicados preservando ordem.
        dedup = []
        seen = set()
        for item in criteria["must_contains"]:
            k = self._norm(item)
            if not k or k in seen:
                continue
            seen.add(k)
            dedup.append(item)
        criteria["must_contains"] = dedup
        criteria["has_any"] = bool(dedup)
        criteria["label"] = ", ".join(dict.fromkeys(criteria["label_parts"]))
        return criteria

    def _record_matches(self, record: dict[str, Any], criteria: dict[str, Any]) -> bool:
        text_norm = self._norm(record.get("text", ""))
        for item in criteria.get("must_contains") or []:
            item_norm = self._norm(item)
            if not item_norm:
                continue
            # Para mês 2025-10/10-2025, aceita qualquer variante no registro.
            if re.fullmatch(r"20\d{2}-\d{2}", item_norm):
                yyyy, mm = item_norm.split("-")
                if item_norm not in text_norm and f"{mm}-{yyyy}" not in text_norm and f"{mm}/{yyyy}" not in text_norm:
                    return False
            elif re.fullmatch(r"\d{2}-20\d{2}", item_norm):
                mm, yyyy = item_norm.split("-")
                if item_norm not in text_norm and f"{yyyy}-{mm}" not in text_norm:
                    return False
            else:
                if item_norm not in text_norm:
                    return False
        return True

    def _extract_main_code(self, text: str) -> str | None:
        m = re.search(r"\b(CHG\d{5,}|INC\d{5,}|REQ\d{4,}|RITM\d{4,}|TASK\d{4,})\b", text, flags=re.IGNORECASE)
        return m.group(1).upper() if m else None

    def _summarize_record(self, code: str, text: str) -> dict[str, str]:
        def first(patterns: list[str]) -> str:
            for pat in patterns:
                m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    return re.sub(r"\s+", " ", m.group(1).strip())[:180]
            return ""
        return {
            "numero": code,
            "tipo": first([r"\bTipo\s*[:\-]\s*(.+?)(?=\s+(?:Estado|Status|Data|IC Impactado|Grupo de atribui|Solicita|Aberta por)\s*[:\-]|$)"]),
            "estado": first([r"\bEstado\s*[:\-]\s*(.+?)(?=\s+(?:Status|Data|IC Impactado|Grupo de atribui|Tipo|Solicita|Aberta por)\s*[:\-]|$)"]),
            "grupo_atribuicao": first([r"\bGrupo de atribui[cç][aã]o\s*[:\-]\s*(.+?)(?=\s+(?:Tipo de Indisponibilidade|Tipo|Estado|Status|Data|IC Impactado|Solicita|Aberta por)\s*[:\-]|$)"]),
            "descricao": first([r"Descri[cç][aã]o resumida\s*[:\-]?\s*(.+?)(?=\s+(?:Tipo de teste|Plano de teste|Risco|$))", r"## Conteúdo do artigo\s*(.+?)($|\n##)"]),
        }

    def _format_deterministic_count_answer(self, question: str, label: str, count: int, rows: list[dict[str, str]], criteria: dict[str, Any], scanned: int) -> str:
        lines = [
            "## Resposta direta",
            "",
            f"Foram encontrados **{count}** registro(s) correspondente(s) ao critério informado.",
            "",
            "## Critérios aplicados",
        ]
        for item in criteria.get("must_contains") or []:
            lines.append(f"- `{item}`")
        lines.extend(["", f"Registros/textos avaliados no case: **{scanned}**."])
        if rows:
            lines.extend(["", "## Registros encontrados"])
            for idx, row in enumerate(rows, start=1):
                detail = " – ".join([v for v in [row.get("tipo"), row.get("estado"), row.get("grupo_atribuicao")] if v])
                lines.append(f"{idx}. **{row.get('numero')}**" + (f" – {detail}" if detail else ""))
        lines.extend(["", "> Resultado gerado por varredura determinística dos documentos indexados do case, não por aproximação semântica."])
        return "\n".join(lines)

    def _format_deterministic_list_answer(self, question: str, label: str, rows: list[dict[str, str]], criteria: dict[str, Any], scanned: int) -> str:
        lines = ["## Registros encontrados", "", f"Total listado: **{len(rows)}**.", "", "## Critérios aplicados"]
        for item in criteria.get("must_contains") or []:
            lines.append(f"- `{item}`")
        if rows:
            lines.extend(["", "## Lista"])
            for idx, row in enumerate(rows, start=1):
                detail = " – ".join([v for v in [row.get("tipo"), row.get("estado"), row.get("grupo_atribuicao")] if v])
                lines.append(f"{idx}. **{row.get('numero')}**" + (f" – {detail}" if detail else ""))
        lines.extend(["", f"Registros/textos avaliados no case: **{scanned}**."])
        return "\n".join(lines)

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
            "excerpt": json.dumps(execution, ensure_ascii=False, default=str)[:6000],
            "metadata": {"source": "tabular", "source_type": table.get("source_type")},
        }]

    def _merge_evidences(self, file_evidences, case_evidences, tabular_evidences, focus: str) -> list[dict[str, Any]]:
        if focus in {"case_upload_first", "hybrid_case_first"}:
            merged = (file_evidences or []) + (case_evidences or []) + (tabular_evidences or [])
        else:
            merged = (tabular_evidences or []) + (file_evidences or []) + (case_evidences or [])
        out = []
        seen = set()
        for e in merged:
            key = (str(e.get("filename")), str(e.get("chunk_id")), str(e.get("type")))
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out[:30]

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
        ])[:24000]
        tabular_blob = ""
        if tabular_attempt:
            tabular_blob = "\n\nResultado tabular auxiliar:\n" + json.dumps(tabular_attempt.get("technical", tabular_attempt), ensure_ascii=False, indent=2, default=str)[:7000]

        system_prompt = """
Você é um analista sênior do GABBI. Responda em português do Brasil.

REGRAS OBRIGATÓRIAS:
1. Use SOMENTE as evidências fornecidas abaixo.
2. Se a evidência existir, responda exatamente o que está nela; não use aproximações.
3. Nunca diga que não tem acesso a arquivo quando houver evidência do tipo direct_file_context ou case_upload.
4. Para números, contagens e listas: não estime. Se não houver varredura determinística, diga que os números dependem das evidências disponíveis.
5. Para perguntas sobre arquivo/anexo/documento, priorize o conteúdo do upload do case.
6. A base treinada/PostgreSQL complementa, mas não substitui o arquivo anexado.
7. Não invente registros, IDs, quantidades, filtros ou conclusões.
8. Se a informação não estiver nas evidências, diga exatamente que não foi encontrada nas evidências disponíveis.
9. Informe de forma curta quais fontes foram usadas.
""".strip()
        if mode == "executive":
            system_prompt += "\nPriorize linguagem executiva, objetiva e orientada à decisão."
        elif mode == "analytical":
            system_prompt += "\nPriorize análise detalhada, riscos, regras e exceções."
        else:
            system_prompt += "\nPriorize precisão técnica."

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
        for e in evidences[:8]:
            lines.append(f"### {e.get('filename')}")
            lines.append(str(e.get("excerpt", ""))[:1200])
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

    def _extract_requested_filenames(self, question: str) -> list[str]:
        names = []
        for m in re.findall(r"([\w\-. ]+\.(?:txt|csv|xlsx|xls|pdf|docx|pptx|md|markdown))", question, flags=re.IGNORECASE):
            names.append(self._norm(m.strip()))
        return names

    @staticmethod
    def _norm(value: Any) -> str:
        text = "" if value is None else str(value)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
        return " ".join(text.split())
