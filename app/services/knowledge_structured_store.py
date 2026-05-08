from __future__ import annotations

import os
import re
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

try:
    import duckdb
    HAS_DUCKDB = True
except Exception:
    duckdb = None
    HAS_DUCKDB = False


def _norm(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
    return " ".join(text.split())


class KnowledgeStructuredStore:
    """DuckDB analítico interno do Nexus para artigos sincronizados do Gabbi.

    Esta camada é propositalmente separada do RAG. Ela serve para perguntas que exigem
    contagem, listagem, distinct e filtros. O RAG continua sendo usado para perguntas
    explicativas/semânticas.
    """

    COUNT_TERMS = {"quantos", "quantas", "quantidade", "qtd", "qtde", "total", "totais", "contar", "conte", "numero", "número"}
    LIST_TERMS = {"liste", "listar", "lista", "quais", "mostre", "exiba", "relacione"}
    DETAIL_TERMS = {"detalhe", "detalhar", "explique", "explicar", "resuma", "resumir", "sobre"}
    GROUP_TERMS = {"agrupe", "agrupar", "distribuicao", "distribuição", "por"}

    def __init__(self, db_path: str | None = None):
        default_path = os.getenv("GABBI_NEXUS_DUCKDB_PATH") or os.getenv("DUCKDB_PATH") or "app/data/gabbi_nexus.duckdb"
        self.db_path = Path(db_path or default_path)
        self.enabled = HAS_DUCKDB
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "reason": "duckdb_not_installed", "install": "pip install duckdb"}
        try:
            with self._connect() as con:
                total = con.execute("SELECT COUNT(*) FROM knowledge_articles_structured").fetchone()[0]
                cases = con.execute("SELECT COUNT(DISTINCT case_id) FROM knowledge_articles_structured").fetchone()[0]
            return {"enabled": True, "db_path": str(self.db_path), "rows": int(total), "cases": int(cases)}
        except Exception as exc:
            return {"enabled": False, "db_path": str(self.db_path), "error": str(exc)}

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("DuckDB não está instalado. Instale com: pip install duckdb")
        return duckdb.connect(str(self.db_path))

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
                    article_updated_on TEXT,
                    article_text TEXT,
                    raw_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_case ON knowledge_articles_structured(case_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_numero ON knowledge_articles_structured(numero)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_tipo_mes ON knowledge_articles_structured(codigo_tipo, mes)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_kas_grupo ON knowledge_articles_structured(grupo_atribuicao)")

    def replace_case_rows(self, case_id: str, rows: list[dict[str, Any]], knowledge_version: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("DuckDB não está instalado. Instale com: pip install duckdb")
        if not case_id:
            raise ValueError("case_id é obrigatório")
        self._ensure_schema()
        normalized = [self._normalize_row(case_id, row, knowledge_version) for row in rows or []]
        with self._connect() as con:
            con.execute("DELETE FROM knowledge_articles_structured WHERE case_id = ?", [case_id])
            if normalized:
                cols = list(normalized[0].keys())
                placeholders = ", ".join(["?"] * len(cols))
                sql = f"INSERT INTO knowledge_articles_structured ({', '.join(cols)}) VALUES ({placeholders})"
                con.executemany(sql, [[row.get(c) for c in cols] for row in normalized])
        return {"success": True, "case_id": case_id, "rows_received": len(rows or []), "rows_saved": len(normalized), "knowledge_version": knowledge_version}

    def _normalize_row(self, case_id: str, row: dict[str, Any], knowledge_version: str | None) -> dict[str, Any]:
        fields = [
            "case_id", "knowledge_version", "agent_key", "agent_name", "project_key", "project_id", "project_name",
            "topic_id", "topic_ref_id", "topic_name", "topic_description", "article_id", "article_ref_id", "source_id",
            "codigo_tipo", "codigo_principal", "numero", "transacao_z", "mes", "tipo", "estado", "status",
            "grupo_atribuicao", "ic_impactado", "canal", "categoria", "prioridade", "data_inicio_planejada",
            "data_termino_planejada", "article_updated_on", "article_text", "raw_json"
        ]
        out = {k: self._safe_text(row.get(k)) for k in fields}
        out["case_id"] = case_id
        out["knowledge_version"] = self._safe_text(row.get("knowledge_version") or knowledge_version)
        out["raw_json"] = json.dumps(row, ensure_ascii=False, default=str)
        # Normalizações essenciais para analytics.
        out["codigo_tipo"] = out["codigo_tipo"].upper()
        if not out["codigo_tipo"]:
            n = (out["numero"] or out["codigo_principal"]).upper()
            if n.startswith("CHG"):
                out["codigo_tipo"] = "CHG"
            elif n.startswith("INC"):
                out["codigo_tipo"] = "INC"
            elif n.startswith("Z"):
                out["codigo_tipo"] = "TRANSACAO_Z"
        if not out["numero"]:
            out["numero"] = out["codigo_principal"] or out["transacao_z"]
        out["numero"] = out["numero"].upper()
        out["codigo_principal"] = (out["codigo_principal"] or out["numero"]).upper()
        out["mes"] = self._normalize_month(out["mes"] or out["topic_name"] or out["topic_description"] or out["article_text"])
        return out

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _normalize_month(self, value: str) -> str:
        text = str(value or "")
        m = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", text)
        if m:
            return f"{m.group(1)}-{m.group(2).zfill(2)}"
        m = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", text)
        if m:
            return f"{m.group(2)}-{m.group(1).zfill(2)}"
        m = re.search(r"\b(0?[1-9]|1[0-2])\s+de\s+(20\d{2})\b", _norm(text))
        if m:
            return f"{m.group(2)}-{m.group(1).zfill(2)}"
        months = {"janeiro":"01","jan":"01","fevereiro":"02","fev":"02","marco":"03","março":"03","mar":"03","abril":"04","abr":"04","maio":"05","junho":"06","jun":"06","julho":"07","jul":"07","agosto":"08","ago":"08","setembro":"09","set":"09","outubro":"10","out":"10","novembro":"11","nov":"11","dezembro":"12","dez":"12"}
        nt = _norm(text)
        y = re.search(r"\b(20\d{2})\b", nt)
        if y:
            for name, mm in months.items():
                if re.search(rf"\b{re.escape(_norm(name))}\b", nt):
                    return f"{y.group(1)}-{mm}"
        return ""

    def answer_question(self, case_id: str, question: str, chat_history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        # Perguntas documentais com código exato devem buscar primeiro o registro exato no DuckDB.
        # Se o registro não existir na base estruturada, cai para RAG/detalhamento textual.
        if self._is_exact_code_detail_question(question):
            detail = self._execute_exact_code_detail(case_id, question)
            if detail:
                return detail
            return None

        intent = self._detect_intent(question)
        if intent not in {"count", "list", "group", "distinct"}:
            return None

        criteria = self._extract_criteria(question, chat_history or [], intent=intent)

        # Só responde se a pergunta tiver cara de base estruturada.
        if not self._is_knowledge_question(question, criteria):
            return None
        try:
            result = self._execute(case_id, intent, criteria)
        except Exception as exc:
            return {"route": "knowledge_structured_duckdb_error", "answer_text": "", "summary": "", "technical": {"error": str(exc), "criteria": criteria}, "fallback_to_rag": True}
        if result is None:
            return None
        return result

    def _is_exact_code_detail_question(self, question: str) -> bool:
        q = _norm(question)
        has_code = re.search(r"\b(?:CHG|INC)\d{5,}\b", question or "", flags=re.IGNORECASE) is not None
        wants_detail = bool(set(q.split()) & self.DETAIL_TERMS)
        return bool(has_code and wants_detail)

    def _is_followup_question(self, question: str) -> bool:
        q = _norm(question)
        return q.startswith(("e ", "e quant", "e quais", "e liste", "e listar", "e com ", "e no ", "e na ", "e em "))

    def _detect_intent(self, question: str) -> str:
        q = _norm(question)
        tokens = set(q.split())
        if any(x in q for x in ["grupos de atribuicao", "grupos de atribuição", "valores de grupo", "quais grupos"]):
            return "distinct"
        if tokens & self.COUNT_TERMS:
            return "count"
        if any(m in q for m in ["agrupe", "agrupar", "distribuicao", "distribuição", " por "]):
            return "group"
        if tokens & self.LIST_TERMS:
            return "list"
        return "describe"

    def _is_knowledge_question(self, question: str, criteria: dict[str, Any]) -> bool:
        q = _norm(question)
        if criteria.get("codigo_tipo") in {"CHG", "INC"}:
            return True
        if any(t in q for t in ["change", "changes", "chg", "incidente", "incidentes", "inc", "grupo de atribuicao", "grupo de atribuição", "ic impactado", "estado", "status"]):
            return True
        return False

    def _extract_criteria(self, question: str, history: list[dict[str, Any]], intent: str | None = None) -> dict[str, Any]:
        # Regra central: a pergunta atual é soberana.
        # Histórico só é usado para herdar escopo alto (codigo_tipo/mes) em follow-up iniciado com "e ...".
        raw_current = question or ""
        q_current = _norm(raw_current)
        c: dict[str, Any] = {}

        # Código exato na pergunta atual zera qualquer contexto anterior.
        exact_code = self._extract_exact_code(raw_current)
        if exact_code:
            c["numero"] = exact_code
            if exact_code.startswith("CHG"):
                c["codigo_tipo"] = "CHG"
            elif exact_code.startswith("INC"):
                c["codigo_tipo"] = "INC"

        # Entidade da pergunta atual.
        if re.search(r"\bchg\b|\bchange\b|\bchanges\b|\bmudanca\b|\bmudancas\b", q_current):
            c["codigo_tipo"] = "CHG"
        if re.search(r"\binc\b|\bincidente\b|\bincidentes\b", q_current):
            c["codigo_tipo"] = "INC"

        month_current = self._extract_month(raw_current)
        if month_current:
            c["mes"] = month_current

        # Valores explícitos só da pergunta atual.
        g = self._extract_explicit_value(raw_current, r"grupo\s+de\s+atribui[cç][aã]o")
        if g and intent != "distinct":
            c["grupo_atribuicao"] = g

        estado = self._extract_explicit_value(raw_current, r"estado|status")
        if estado and intent != "distinct":
            c["estado"] = estado

        tipo = self._extract_explicit_value(raw_current, r"tipo")
        if tipo and tipo.lower() not in {"chg", "inc", "change", "changes"}:
            c["tipo"] = tipo

        ic = self._extract_explicit_value(raw_current, r"ic\s+impactado|ic")
        if ic:
            c["ic_impactado"] = ic

        # Herança controlada: apenas em follow-up explícito "e ..." e apenas escopo alto.
        if self._is_followup_question(raw_current):
            inherited = self._extract_high_scope_from_history(history)
            for key in ("codigo_tipo", "mes"):
                if not c.get(key) and inherited.get(key):
                    c[key] = inherited[key]

        return c

    def _extract_exact_code(self, text: str) -> str:
        m = re.search(r"\b((?:CHG|INC)\d{5,})\b", text or "", flags=re.IGNORECASE)
        return m.group(1).upper() if m else ""

    def _extract_high_scope_from_history(self, history: list[dict[str, Any]]) -> dict[str, Any]:
        inherited: dict[str, Any] = {}
        for item in reversed(history or []):
            if not isinstance(item, dict):
                continue
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

    def _extract_month(self, text: str) -> str:
        raw = text or ""
        m = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", raw)
        if m:
            return f"{m.group(1)}-{m.group(2).zfill(2)}"
        m = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", raw)
        if m:
            return f"{m.group(2)}-{m.group(1).zfill(2)}"
        m = re.search(r"\b(0?[1-9]|1[0-2])\s+de\s+(20\d{2})\b", _norm(raw))
        if m:
            return f"{m.group(2)}-{m.group(1).zfill(2)}"
        return ""

    def _extract_explicit_value(self, raw: str, field_regex: str) -> str:
        patterns = [
            rf"(?:{field_regex})\s*[:=]\s*([^\n\?;,.]+)",
            rf"(?:com|de|do|da|dos|das)\s+(?:{field_regex})\s+([^\n\?;,.]+)",
        ]
        for pat in patterns:
            m = re.search(pat, raw, flags=re.IGNORECASE)
            if m:
                value = m.group(1).strip().strip(" .;,'\"“”")
                value = re.split(r"\s+(?:no|na|em|com|e)\s+", value, flags=re.IGNORECASE)[0].strip()
                bad = {"quantas", "quantos", "temos", "existe", "existem", "lista", "liste", "total"}
                if value and _norm(value) not in bad:
                    return value
        return ""


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
                SELECT
                    numero, codigo_tipo, mes, tipo, estado, status, grupo_atribuicao, ic_impactado,
                    canal, categoria, prioridade, data_inicio_planejada, data_termino_planejada,
                    topic_name, project_name, source_id, article_ref_id, article_text
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

        lines: list[str] = []
        if len(rows) == 1:
            r = rows[0]
            fields = {
                "Número": r[0],
                "Tipo de código": r[1],
                "Mês": r[2],
                "Tipo": r[3],
                "Estado": r[4] or r[5],
                "Grupo de atribuição": r[6],
                "IC impactado": r[7],
                "Canal": r[8],
                "Categoria": r[9],
                "Prioridade": r[10],
                "Data de início planejada": r[11],
                "Data de término planejada": r[12],
                "Tópico": r[13],
                "Projeto": r[14],
                "Fonte": r[15] or r[16],
            }
            lines.append(f"## Detalhamento da {code}")
            lines.append("")
            for label, value in fields.items():
                if value:
                    lines.append(f"- **{label}:** {value}")
            article_text = str(r[17] or "").strip()
            if article_text:
                lines.append("")
                lines.append("## Conteúdo do artigo")
                lines.append(article_text[:6000])
        else:
            lines.append(f"## Registros encontrados para {code}")
            for r in rows:
                details = " – ".join([str(x) for x in [r[2], r[3], r[4] or r[5], r[6]] if x])
                lines.append(f"- **{r[0]}**" + (f" – {details}" if details else ""))

        return self._response(
            case_id=case_id,
            answer="\n".join(lines),
            query_type="detail",
            criteria={"numero": code, "codigo_tipo": "CHG" if code.startswith("CHG") else "INC" if code.startswith("INC") else ""},
            technical={"rows_in_case": int(total_case or 0), "matched_rows": len(rows), "sql_mode": "exact_code_detail"},
        )

    def _execute(self, case_id: str, intent: str, criteria: dict[str, Any]) -> dict[str, Any] | None:
        where, params = self._build_where(case_id, criteria)
        with self._connect() as con:
            total_case = con.execute("SELECT COUNT(*) FROM knowledge_articles_structured WHERE case_id = ?", [case_id]).fetchone()[0]
            if int(total_case) == 0:
                return None
            if intent == "count":
                sql = "SELECT COUNT(DISTINCT COALESCE(NULLIF(numero,''), article_id)) FROM knowledge_articles_structured " + where
                count = con.execute(sql, params).fetchone()[0]
                return self._response(case_id, str(int(count or 0)), "count", criteria, {"rows_in_case": int(total_case), "sql_mode": "count_distinct_numero"})
            if intent == "distinct":
                # Atualmente usado principalmente para grupos de atribuição.
                field = "grupo_atribuicao"
                if "estado" in _norm(json.dumps(criteria, ensure_ascii=False)):
                    field = "estado"
                sql = f"SELECT DISTINCT {field} FROM knowledge_articles_structured " + where + f" AND NULLIF({field}, '') IS NOT NULL ORDER BY {field}"
                rows = [r[0] for r in con.execute(sql, params).fetchall() if r[0]]
                answer = "\n".join(f"- {v}" for v in rows) if rows else "Nenhum valor encontrado."
                return self._response(case_id, answer, "distinct", criteria, {"rows_in_case": int(total_case), "field": field, "distinct_count": len(rows)})
            if intent == "group":
                field = "grupo_atribuicao"
                sql = f"SELECT {field}, COUNT(DISTINCT COALESCE(NULLIF(numero,''), article_id)) total FROM knowledge_articles_structured " + where + f" GROUP BY {field} ORDER BY total DESC, {field}"
                rows = con.execute(sql, params).fetchall()
                answer = "\n".join(f"- {r[0] or '(vazio)'}: {int(r[1])}" for r in rows) if rows else "Nenhum registro encontrado."
                return self._response(case_id, answer, "group", criteria, {"rows_in_case": int(total_case), "group_by": field, "groups": len(rows)})
            if intent == "list":
                sql = "SELECT DISTINCT numero, tipo, estado, grupo_atribuicao, mes FROM knowledge_articles_structured " + where + " ORDER BY numero LIMIT 500"
                rows = con.execute(sql, params).fetchall()
                if not rows:
                    answer = "Nenhum registro encontrado."
                else:
                    lines = []
                    for r in rows:
                        details = " – ".join([x for x in [r[1], r[2], r[3], r[4]] if x])
                        lines.append(f"- {r[0]}" + (f" – {details}" if details else ""))
                    answer = "\n".join(lines)
                return self._response(case_id, answer, "list", criteria, {"rows_in_case": int(total_case), "listed": len(rows)})
        return None

    def _build_where(self, case_id: str, criteria: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses = ["case_id = ?"]
        params: list[Any] = [case_id]
        exact_cols = ["codigo_tipo", "mes"]
        contains_cols = ["grupo_atribuicao", "estado", "status", "tipo", "ic_impactado", "canal", "categoria", "numero"]
        for col in exact_cols:
            if criteria.get(col):
                clauses.append(f"UPPER({col}) = UPPER(?)")
                params.append(criteria[col])
        for col in contains_cols:
            if criteria.get(col):
                clauses.append(f"{col} ILIKE ?")
                params.append(f"%{criteria[col]}%")
        return "WHERE " + " AND ".join(clauses), params

    def _response(self, case_id: str, answer: str, query_type: str, criteria: dict[str, Any], technical: dict[str, Any]) -> dict[str, Any]:
        return {
            "route": "knowledge_structured_duckdb",
            "query_type": query_type,
            "answer_text": answer,
            "summary": answer,
            "evidence_files": ["DuckDB:knowledge_articles_structured"],
            "technical": {"case_id": case_id, "criteria": criteria, **technical},
            "sources": {"deterministic": True, "engine": "duckdb", "table": "knowledge_articles_structured"},
        }
