from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    import duckdb
    HAS_DUCKDB = True
except Exception:
    duckdb = None
    HAS_DUCKDB = False


def _safe(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _norm(value: Any) -> str:
    text = _safe(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
    return " ".join(text.split())


class KnowledgeStructuredStore:
    """Fonte oficial determinística para perguntas analíticas de CHG/INC.

    Regras do hotfix:
    - COUNT/LIST/GROUP/DISTINCT nunca dependem de RAG.
    - Contagem é por código distinto (`numero`), não por chunk/linha textual.
    - Filtros críticos suportados: mês, dia, IC Impactado, grupo, APP/ECOMM,
      causado pela mudança, Aura/WhatsApp e códigos CHG/INC.
    - Follow-up herda escopo alto e lista anterior via `technical.criteria/results`.
    """

    COUNT_TERMS = {"quantos", "quantas", "quantidade", "qtd", "qtde", "total", "totais", "contar", "conte", "numero", "número"}
    LIST_TERMS = {"liste", "listar", "lista", "quais", "qual", "mostre", "exiba", "relacione", "traga", "códigos", "codigos"}
    GROUP_TERMS = {"agrupe", "agrupar", "distribuicao", "distribuição", "distribuir", "por"}
    DETAIL_TERMS = {"detalhe", "detalhar", "explique", "explicar", "resuma", "resumir", "sobre", "fale"}

    DISTINCT_CAPABLE_FIELDS = {
        "grupo_atribuicao", "estado", "tipo", "ic_impactado", "canal", "categoria", "prioridade", "mes", "codigo_tipo", "servico"
    }

    COLUMNS = [
        "case_id", "knowledge_version", "agent_key", "agent_name", "project_key", "project_id", "project_name",
        "topic_id", "topic_ref_id", "topic_name", "topic_description", "article_id", "article_ref_id", "source_id",
        "codigo_tipo", "codigo_principal", "numero", "transacao_z", "mes", "tipo", "estado", "status",
        "grupo_atribuicao", "ic_impactado", "canal", "categoria", "prioridade", "data_inicio_planejada",
        "data_termino_planejada", "aberto", "resolvido", "encerrado", "causado_pela_mudanca", "is_app", "is_ecomm",
        "servico", "descricao_resumida", "article_updated_on", "article_text", "raw_json"
    ]

    def __init__(self, db_path: str | None = None):
        default_path = os.getenv("GABBI_NEXUS_DUCKDB_PATH") or os.getenv("DUCKDB_PATH") or "app/data/gabbi_nexus.duckdb"
        self.db_path = Path(db_path or default_path)
        self.enabled = HAS_DUCKDB
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("DuckDB não está instalado. Instale com: pip install duckdb")
        return duckdb.connect(str(self.db_path))

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "reason": "duckdb_not_installed", "install": "pip install duckdb"}
        try:
            self._ensure_schema()
            with self._connect() as con:
                total = con.execute("SELECT COUNT(*) FROM knowledge_articles_structured").fetchone()[0]
                cases = con.execute("SELECT COUNT(DISTINCT case_id) FROM knowledge_articles_structured").fetchone()[0]
                chg = con.execute("SELECT COUNT(DISTINCT numero) FROM knowledge_articles_structured WHERE codigo_tipo='CHG'").fetchone()[0]
                inc = con.execute("SELECT COUNT(DISTINCT numero) FROM knowledge_articles_structured WHERE codigo_tipo='INC'").fetchone()[0]
            return {"enabled": True, "db_path": str(self.db_path), "rows": int(total), "cases": int(cases), "changes": int(chg), "incidents": int(inc)}
        except Exception as exc:
            return {"enabled": False, "db_path": str(self.db_path), "error": str(exc)}

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_articles_structured (
                    case_id TEXT,
                    knowledge_version TEXT,
                    agent_key TEXT,
                    agent_name TEXT,
                    project_key TEXT,
                    project_id TEXT,
                    project_name TEXT,
                    topic_id TEXT,
                    topic_ref_id TEXT,
                    topic_name TEXT,
                    topic_description TEXT,
                    article_id TEXT,
                    article_ref_id TEXT,
                    source_id TEXT,
                    codigo_tipo TEXT,
                    codigo_principal TEXT,
                    numero TEXT,
                    transacao_z TEXT,
                    mes TEXT,
                    tipo TEXT,
                    estado TEXT,
                    status TEXT,
                    grupo_atribuicao TEXT,
                    ic_impactado TEXT,
                    canal TEXT,
                    categoria TEXT,
                    prioridade TEXT,
                    data_inicio_planejada TEXT,
                    data_termino_planejada TEXT,
                    aberto TEXT,
                    resolvido TEXT,
                    encerrado TEXT,
                    causado_pela_mudanca TEXT,
                    is_app TEXT,
                    is_ecomm TEXT,
                    servico TEXT,
                    descricao_resumida TEXT,
                    article_updated_on TEXT,
                    article_text TEXT,
                    raw_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Migração leve para bases já criadas em versões anteriores.
            existing = {r[1] for r in con.execute("PRAGMA table_info('knowledge_articles_structured')").fetchall()}
            for col in self.COLUMNS:
                if col not in existing:
                    con.execute(f"ALTER TABLE knowledge_articles_structured ADD COLUMN {col} TEXT")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_case ON knowledge_articles_structured(case_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_numero ON knowledge_articles_structured(numero)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_tipo_mes ON knowledge_articles_structured(codigo_tipo, mes)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_grupo ON knowledge_articles_structured(grupo_atribuicao)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_ic ON knowledge_articles_structured(ic_impactado)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_mes ON knowledge_articles_structured(mes)")

    def replace_case_rows(self, case_id: str, rows: list[dict[str, Any]], knowledge_version: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("DuckDB não está instalado. Instale com: pip install duckdb")
        if not case_id:
            raise ValueError("case_id é obrigatório")
        self._ensure_schema()
        normalized = [self._normalize_row(case_id, row, knowledge_version) for row in rows or []]
        normalized = [r for r in normalized if r.get("codigo_tipo") and r.get("numero")]
        with self._connect() as con:
            con.execute("DELETE FROM knowledge_articles_structured WHERE case_id = ?", [case_id])
            if normalized:
                cols = list(self.COLUMNS)
                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT INTO knowledge_articles_structured ({', '.join(cols)}) VALUES ({placeholders})"
                con.executemany(sql, [[row.get(c, "") for c in cols] for row in normalized])
        return {"success": True, "case_id": case_id, "rows_received": len(rows or []), "rows_saved": len(normalized), "knowledge_version": knowledge_version}

    def _normalize_row(self, case_id: str, row: dict[str, Any], knowledge_version: str | None) -> dict[str, Any]:
        out = {k: _safe(row.get(k)) for k in self.COLUMNS}
        out["case_id"] = case_id
        out["knowledge_version"] = _safe(row.get("knowledge_version") or knowledge_version)
        out["raw_json"] = out.get("raw_json") or json.dumps(row, ensure_ascii=False, default=str)
        out["codigo_tipo"] = out["codigo_tipo"].upper()
        all_text = " ".join([out.get("numero", ""), out.get("codigo_principal", ""), out.get("article_text", "")]).upper()
        if not out["codigo_tipo"]:
            if re.search(r"\bCHG\d{5,}\b", all_text):
                out["codigo_tipo"] = "CHG"
            elif re.search(r"\bINC\d{5,}\b", all_text):
                out["codigo_tipo"] = "INC"
            elif re.search(r"\bZ[A-Z0-9]{2,}\b", all_text):
                out["codigo_tipo"] = "TRANSACAO_Z"
        if not out["numero"]:
            m = re.search(r"\b((?:CHG|INC)\d{5,})\b", all_text, flags=re.IGNORECASE)
            out["numero"] = m.group(1).upper() if m else (out.get("codigo_principal") or out.get("transacao_z"))
        out["numero"] = out["numero"].upper()
        out["codigo_principal"] = (out["codigo_principal"] or out["numero"]).upper()
        out["mes"] = self._normalize_month(out.get("mes") or out.get("aberto") or out.get("data_inicio_planejada") or out.get("topic_name") or out.get("article_text"))
        if not out["status"]:
            out["status"] = out["estado"]
        for b in ["is_app", "is_ecomm"]:
            n = _norm(out.get(b))
            out[b] = "true" if n in {"true", "1", "sim", "yes", "y"} else "false" if n in {"false", "0", "nao", "não", "no", "n"} else out.get(b, "")
        return out

    def _normalize_month(self, value: Any) -> str:
        text = _safe(value)
        m = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", text)
        if m:
            return f"{m.group(1)}-{m.group(2).zfill(2)}"
        m = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", text)
        if m:
            return f"{m.group(2)}-{m.group(1).zfill(2)}"
        nt = _norm(text)
        m = re.search(r"\b(0?[1-9]|1[0-2])\s+de\s+(20\d{2})\b", nt)
        if m:
            return f"{m.group(2)}-{m.group(1).zfill(2)}"
        months = {"janeiro":"01","jan":"01","fevereiro":"02","fev":"02","marco":"03","março":"03","mar":"03","abril":"04","abr":"04","maio":"05","junho":"06","jun":"06","julho":"07","jul":"07","agosto":"08","ago":"08","setembro":"09","set":"09","outubro":"10","out":"10","novembro":"11","nov":"11","dezembro":"12","dez":"12"}
        y = re.search(r"\b(20\d{2})\b", nt)
        if y:
            for name, mm in months.items():
                if re.search(rf"\b{re.escape(_norm(name))}\b", nt):
                    return f"{y.group(1)}-{mm}"
        return ""

    def answer_question(self, case_id: str, question: str, chat_history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        if not self.enabled or not question or not question.strip():
            return None
        self._ensure_schema()
        qn = _norm(question)
        # Follow-up: "quais os códigos?" deve retornar o último result_set se existir.
        followup_codes = self._last_codes_from_history(chat_history or [])
        if followup_codes and re.search(r"\b(quais|liste|listar|mostre|codigos|códigos)\b", qn) and not re.search(r"\b(CHG|INC)\d{5,}\b", question, re.I):
            answer = "\n".join(f"- {c}" for c in followup_codes)
            return self._response(case_id, answer, "list", {"from_previous_result": True}, {"result_codes": followup_codes, "sql_mode": "previous_result_set"})

        if self._is_exact_code_detail_question(question):
            detail = self._execute_exact_code_detail(case_id, question)
            if detail:
                return detail
            return None

        intent = self._detect_intent(question)
        if intent not in {"count", "list", "group", "distinct"}:
            return None
        criteria = self._extract_criteria(question, chat_history or [], intent=intent)
        if not self._is_knowledge_question(question, criteria, intent):
            return None
        try:
            return self._execute(case_id, intent, criteria)
        except Exception as exc:
            # Não deixa RAG chutar analytics quando existe clara intenção analítica.
            return self._response(
                case_id,
                "Não consegui executar a consulta analítica estruturada com segurança. Verifique a ingestão estruturada DuckDB e os filtros informados.",
                intent,
                criteria,
                {"error": str(exc), "fallback_blocked": True, "sql_mode": "structured_error"},
            )

    def _detect_intent(self, question: str) -> str:
        q = _norm(question)
        tokens = set(q.split())
        if self._detect_distinct_field(question) and (tokens & self.LIST_TERMS or "existentes" in tokens or "distintos" in tokens or "distintas" in tokens):
            return "distinct"
        if tokens & self.COUNT_TERMS:
            return "count"
        if any(t in q for t in ["agrupe", "agrupar", "distribuicao", "distribuição", "distribuir"]):
            return "group"
        if " por " in q and (tokens & self.COUNT_TERMS or "total" in tokens):
            return "group"
        if tokens & self.LIST_TERMS:
            return "list"
        return "describe"

    def _is_knowledge_question(self, question: str, criteria: dict[str, Any], intent: str) -> bool:
        q = _norm(question)
        if criteria.get("codigo_tipo") in {"CHG", "INC", "TRANSACAO_Z"}:
            return True
        if criteria.get("mes") and intent in {"count", "list", "group", "distinct"}:
            return True
        return any(t in q for t in [
            "change", "changes", "chg", "mudanca", "mudancas", "incidente", "incidentes", "inc",
            "grupo de atribuicao", "grupo de atribuição", "ic impactado", "estado", "status", "canal", "categoria",
            "prioridade", "tipo", "app", "ecomm", "aura", "whatsapp", "causado pela mudanca", "causado pela mudança"
        ])

    def _detect_distinct_field(self, question: str) -> str:
        q = _norm(question)
        if "grupo" in q:
            return "grupo_atribuicao"
        if "ic impactado" in q or re.search(r"\bics?\b", q):
            return "ic_impactado"
        if "canal" in q or "canais" in q:
            return "canal"
        if "servico" in q or "serviço" in q:
            return "servico"
        if "categoria" in q:
            return "categoria"
        if "prioridade" in q:
            return "prioridade"
        if "estado" in q or "status" in q or "situacao" in q:
            return "estado"
        if re.search(r"\btipos?\b", q):
            return "tipo"
        if re.search(r"\bmes(?:es)?\b|\bperiodos?\b", q):
            return "mes"
        return ""

    def _detect_group_field(self, question: str) -> str:
        q = _norm(question)
        tail = q.split(" por ", 1)[1] if " por " in q else q
        for needle, field in [("grupo", "grupo_atribuicao"), ("estado", "estado"), ("status", "estado"), ("tipo", "tipo"), ("ic", "ic_impactado"), ("canal", "canal"), ("categoria", "categoria"), ("prioridade", "prioridade"), ("mes", "mes"), ("servico", "servico")]:
            if needle in tail:
                return field
        return "grupo_atribuicao"

    def _extract_criteria(self, question: str, history: list[dict[str, Any]], intent: str | None = None) -> dict[str, Any]:
        raw = question or ""
        q = _norm(raw)
        c: dict[str, Any] = {}
        code = self._extract_exact_code(raw)
        if code:
            if code.startswith("CHG") and any(t in q for t in ["referencia", "referência", "causado pela", "fazem referencia", "faz referência"]):
                c["causado_pela_mudanca"] = code
            else:
                c["numero"] = code
            c["codigo_tipo"] = "CHG" if code.startswith("CHG") else "INC" if code.startswith("INC") else ""

        if re.search(r"\bchg\b|\bchange\b|\bchanges\b|\bmudanca\b|\bmudancas\b", q):
            c["codigo_tipo"] = "CHG"
        elif re.search(r"\binc\b|\bincidente\b|\bincidentes\b", q):
            c["codigo_tipo"] = "INC"

        month = self._extract_month(raw)
        if month:
            c["mes"] = month

        day = self._extract_day(raw)
        if day:
            c["dia"] = day

        if re.search(r"\bapp\b|aplicativo vivo", q):
            c["is_app"] = "true"
        if re.search(r"ecomm|e-commerce|ecommerce", q):
            c["is_ecomm"] = "true"
        if "aura" in q and "whatsapp" in q:
            c["text_search"] = "aura whatsapp"
        elif "whatsapp" in q:
            c["text_search"] = "whatsapp"
        elif "aura" in q:
            c["text_search"] = "aura"

        distinct_field = self._detect_distinct_field(raw) if intent == "distinct" else ""
        for col, regex in [
            ("grupo_atribuicao", r"grupo\s+de\s+atribui[cç][aã]o|grupo"),
            ("estado", r"estado|status|situacao|situação"),
            ("tipo", r"tipo"),
            ("ic_impactado", r"ic\s+impactado|ic"),
            ("canal", r"canal"),
            ("categoria", r"categoria"),
            ("prioridade", r"prioridade"),
        ]:
            if distinct_field == col:
                continue
            val = self._extract_explicit_value(raw, regex)
            if val:
                if col == "tipo" and _norm(val) in {"chg", "inc", "change", "changes", "mudanca", "mudancas", "registro", "registros"}:
                    continue
                c[col] = val

        # Herança controlada para follow-ups: "e ...", "quais os códigos", "alguma dessas".
        if self._is_followup_question(raw):
            inherited = self._extract_high_scope_from_history(history)
            for key in ("codigo_tipo", "mes", "dia"):
                if not c.get(key) and inherited.get(key):
                    c[key] = inherited[key]
            if "alguma dessas" in q or "dessas" in q or "desses" in q:
                codes = self._last_codes_from_history(history)
                if codes:
                    c["numero_in"] = codes
        return c

    def _is_followup_question(self, question: str) -> bool:
        q = _norm(question)
        return q.startswith(("e ", "e quant", "e quais", "e liste", "e listar", "e com ", "e no ", "e na ", "e em ")) or any(x in q for x in ["dessas", "desses", "destas", "destes", "alguma dessas", "quais os codigos", "quais os códigos"])

    def _extract_exact_code(self, text: str) -> str:
        m = re.search(r"\b((?:CHG|INC)\d{5,})\b", text or "", flags=re.IGNORECASE)
        return m.group(1).upper() if m else ""

    def _extract_day(self, text: str) -> str:
        q = _norm(text)
        m = re.search(r"\bdia\s+(\d{1,2})\b", q)
        if m and 1 <= int(m.group(1)) <= 31:
            return m.group(1).zfill(2)
        m = re.search(r"\babertas?\s+no\s+dia\s+(\d{1,2})\b", q)
        if m and 1 <= int(m.group(1)) <= 31:
            return m.group(1).zfill(2)
        return ""

    def _extract_month(self, text: str) -> str:
        return self._normalize_month(text)

    def _extract_explicit_value(self, raw: str, field_regex: str) -> str:
        patterns = [
            rf"(?:{field_regex})\s*[:=]\s*([^\n\?;,.]+)",
            rf"(?:com|de|do|da|dos|das)\s+(?:{field_regex})\s*[:=]?\s+([^\n\?;,.]+)",
        ]
        for pat in patterns:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if not m:
                continue
            value = m.group(1).strip().strip(" .;,'\"“”")
            value = re.split(r"\s+(?:no|na|em|com|e|por)\s+", value, flags=re.IGNORECASE)[0].strip()
            nv = _norm(value)
            bad = {"quantas", "quantos", "temos", "existe", "existem", "lista", "liste", "listar", "todos", "todas", "existentes", "em", "no", "na", "mes", "mês", "periodo", "período"}
            if value and nv not in bad and len(nv) > 1:
                return value
        return ""

    def _extract_high_scope_from_history(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        inherited: dict[str, Any] = {}
        for item in reversed(history or []):
            if not isinstance(item, dict):
                continue
            technical = item.get("technical") or {}
            criteria = technical.get("criteria") if isinstance(technical, dict) else None
            if isinstance(criteria, dict):
                for key in ("codigo_tipo", "mes", "dia"):
                    if not inherited.get(key) and criteria.get(key):
                        inherited[key] = criteria.get(key)
            q = str(item.get("question") or "")
            qn = _norm(q)
            if not inherited.get("codigo_tipo"):
                if re.search(r"\bchg\b|\bchange\b|\bchanges\b|\bmudanca\b|\bmudancas\b", qn):
                    inherited["codigo_tipo"] = "CHG"
                elif re.search(r"\binc\b|\bincidente\b|\bincidentes\b", qn):
                    inherited["codigo_tipo"] = "INC"
            if not inherited.get("mes"):
                m = self._extract_month(q)
                if m:
                    inherited["mes"] = m
            if inherited.get("codigo_tipo") and inherited.get("mes"):
                break
        return inherited

    def _last_codes_from_history(self, history: list[dict[str, Any]]) -> list[str]:
        for item in reversed(history or []):
            if not isinstance(item, dict):
                continue
            technical = item.get("technical") or {}
            if isinstance(technical, dict):
                codes = technical.get("result_codes") or technical.get("codes")
                if isinstance(codes, list) and codes:
                    return [str(c).upper() for c in codes if c]
            text = "\n".join([str(item.get("answer_text") or ""), str(item.get("summary") or "")])
            found = re.findall(r"\b(?:CHG|INC)\d{5,}\b", text, flags=re.IGNORECASE)
            if found:
                out = []
                seen = set()
                for c in found:
                    uc = c.upper()
                    if uc not in seen:
                        seen.add(uc); out.append(uc)
                return out
        return []

    def _is_exact_code_detail_question(self, question: str) -> bool:
        q = _norm(question)
        has_code = re.search(r"\b(?:CHG|INC)\d{5,}\b", question or "", flags=re.IGNORECASE) is not None
        wants_detail = bool(set(q.split()) & self.DETAIL_TERMS)
        return bool(has_code and wants_detail)

    def _build_where(self, case_id: str, criteria: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses = ["case_id = ?"]
        params: list[Any] = [case_id]
        for col in ["codigo_tipo", "mes"]:
            if criteria.get(col):
                clauses.append(f"UPPER({col}) = UPPER(?)")
                params.append(criteria[col])
        for col in ["grupo_atribuicao", "estado", "status", "tipo", "ic_impactado", "canal", "categoria", "prioridade", "numero", "causado_pela_mudanca", "servico"]:
            if criteria.get(col):
                clauses.append(f"{col} ILIKE ?")
                params.append(f"%{criteria[col]}%")
        for col in ["is_app", "is_ecomm"]:
            if criteria.get(col):
                clauses.append(f"LOWER({col}) = LOWER(?)")
                params.append(criteria[col])
        if criteria.get("numero_in"):
            codes = [str(c).upper() for c in criteria.get("numero_in") or [] if c]
            if codes:
                placeholders = ",".join(["?"] * len(codes))
                clauses.append(f"UPPER(numero) IN ({placeholders})")
                params.extend(codes)
        if criteria.get("dia"):
            day = str(criteria["dia"]).zfill(2)
            # Para CHG, usa data_inicio_planejada. Para INC, usa Aberto. Sem tipo, aceita ambos.
            if criteria.get("mes"):
                prefix = f"{criteria['mes']}-{day}"
                clauses.append("(data_inicio_planejada LIKE ? OR aberto LIKE ?)")
                params.extend([prefix + "%", prefix + "%"])
            else:
                clauses.append("(regexp_matches(data_inicio_planejada, ?) OR regexp_matches(aberto, ?))")
                params.extend([rf"^20\d{{2}}-\d{{2}}-{day}", rf"^20\d{{2}}-\d{{2}}-{day}"])
        if criteria.get("text_search"):
            val = "%" + "%".join(criteria["text_search"].split()) + "%"
            clauses.append("(article_text ILIKE ? OR ic_impactado ILIKE ? OR canal ILIKE ? OR descricao_resumida ILIKE ? OR servico ILIKE ?)")
            params.extend([val, val, val, val, val])
        return "WHERE " + " AND ".join(clauses), params

    def _execute_exact_code_detail(self, case_id: str, question: str) -> dict[str, Any] | None:
        code = self._extract_exact_code(question)
        if not code:
            return None
        with self._connect() as con:
            total_case = con.execute("SELECT COUNT(*) FROM knowledge_articles_structured WHERE case_id = ?", [case_id]).fetchone()[0]
            if int(total_case or 0) == 0:
                return None
            rows = con.execute(
                """
                SELECT numero, codigo_tipo, mes, tipo, estado, status, grupo_atribuicao, ic_impactado,
                       canal, categoria, prioridade, data_inicio_planejada, data_termino_planejada,
                       aberto, causado_pela_mudanca, servico, descricao_resumida, topic_name, project_name, source_id, article_ref_id, article_text
                FROM knowledge_articles_structured
                WHERE case_id = ?
                  AND (UPPER(numero) = UPPER(?) OR UPPER(codigo_principal) = UPPER(?) OR article_text ILIKE ?)
                ORDER BY article_ref_id NULLS LAST
                LIMIT 3
                """,
                [case_id, code, code, f"%{code}%"],
            ).fetchall()
        if not rows:
            return None
        lines = [f"## Detalhamento da {code}", ""]
        for r in rows:
            fields = {
                "Número": r[0], "Tipo de código": r[1], "Mês": r[2], "Tipo": r[3], "Estado": r[4] or r[5],
                "Grupo de atribuição": r[6], "IC impactado": r[7], "Canal": r[8], "Categoria": r[9],
                "Prioridade": r[10], "Data de início planejada": r[11], "Data de término planejada": r[12],
                "Aberto": r[13], "Causado pela mudança": r[14], "Serviço": r[15], "Descrição resumida": r[16],
                "Tópico": r[17], "Projeto": r[18], "Fonte": r[19] or r[20],
            }
            for label, value in fields.items():
                if value:
                    lines.append(f"- **{label}:** {value}")
            if len(rows) == 1 and r[21]:
                lines.extend(["", "## Conteúdo do artigo", str(r[21])[:6000]])
        return self._response(case_id, "\n".join(lines), "detail", {"numero": code}, {"rows_in_case": int(total_case or 0), "matched_rows": len(rows), "sql_mode": "exact_code_detail"})

    def _execute(self, case_id: str, intent: str, criteria: dict[str, Any]) -> dict[str, Any] | None:
        criteria = dict(criteria or {})
        distinct_field = criteria.pop("__distinct_field", "") or self._detect_distinct_field(criteria.get("question", "")) or "grupo_atribuicao"
        group_field = criteria.pop("__group_field", "") or "grupo_atribuicao"
        where, params = self._build_where(case_id, criteria)
        with self._connect() as con:
            total_case = con.execute("SELECT COUNT(*) FROM knowledge_articles_structured WHERE case_id = ?", [case_id]).fetchone()[0]
            if int(total_case or 0) == 0:
                return None
            if intent == "count":
                sql = "SELECT COUNT(DISTINCT COALESCE(NULLIF(numero,''), article_id)) FROM knowledge_articles_structured " + where
                count = con.execute(sql, params).fetchone()[0]
                codes = [r[0] for r in con.execute("SELECT DISTINCT numero FROM knowledge_articles_structured " + where + " AND NULLIF(numero,'') IS NOT NULL ORDER BY numero LIMIT 500", params).fetchall()]
                return self._response(case_id, str(int(count or 0)), "count", criteria, {"rows_in_case": int(total_case), "result_codes": codes, "sql_mode": "count_distinct_numero"})
            if intent == "distinct":
                field = self._safe_sql_field(distinct_field, "grupo_atribuicao")
                rows = [str(r[0]).strip() for r in con.execute(f"SELECT DISTINCT {field} FROM knowledge_articles_structured " + where + f" AND NULLIF({field}, '') IS NOT NULL ORDER BY {field}", params).fetchall() if r[0] and str(r[0]).strip()]
                answer = "\n".join(f"- {v}" for v in rows) if rows else "Nenhum valor encontrado."
                return self._response(case_id, answer, "distinct", criteria, {"rows_in_case": int(total_case), "field": field, "distinct_count": len(rows), "sql_mode": "distinct"})
            if intent == "group":
                field = self._safe_sql_field(group_field, "grupo_atribuicao")
                rows = con.execute(f"SELECT {field}, COUNT(DISTINCT COALESCE(NULLIF(numero,''), article_id)) total FROM knowledge_articles_structured " + where + f" GROUP BY {field} ORDER BY total DESC, {field}", params).fetchall()
                answer = "\n".join(f"- {r[0] or '(vazio)'}: {int(r[1])}" for r in rows) if rows else "Nenhum registro encontrado."
                return self._response(case_id, answer, "group", criteria, {"rows_in_case": int(total_case), "group_by": field, "groups": len(rows), "sql_mode": "group_by"})
            if intent == "list":
                rows = con.execute("SELECT DISTINCT numero, tipo, estado, grupo_atribuicao, ic_impactado, mes FROM knowledge_articles_structured " + where + " ORDER BY numero LIMIT 500", params).fetchall()
                codes = [r[0] for r in rows if r[0]]
                if not rows:
                    answer = "Nenhum registro encontrado."
                else:
                    lines = []
                    for r in rows:
                        details = " – ".join([str(x) for x in [r[1], r[2], r[3], r[4], r[5]] if x])
                        lines.append(f"- {r[0]}" + (f" – {details}" if details else ""))
                    answer = "\n".join(lines)
                return self._response(case_id, answer, "list", criteria, {"rows_in_case": int(total_case), "listed": len(rows), "result_codes": codes, "sql_mode": "list_records"})
        return None

    def _safe_sql_field(self, field: str, fallback: str) -> str:
        return field if field in self.DISTINCT_CAPABLE_FIELDS else fallback

    def _response(self, case_id: str, answer: str, query_type: str, criteria: dict[str, Any], technical: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": "knowledge_structured_duckdb",
            "query_type": query_type,
            "answer_text": answer,
            "summary": answer,
            "evidence_files": ["DuckDB:knowledge_articles_structured"],
            "technical": {"case_id": case_id, "criteria": criteria, **(technical or {})},
            "sources": {"deterministic": True, "engine": "duckdb", "table": "knowledge_articles_structured"},
        }
