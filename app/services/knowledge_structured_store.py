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
    text = unicodedata.normalize("NFKD", _safe(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
    return " ".join(text.split())


def _upper(value: Any) -> str:
    return _safe(value).upper().strip()


def _json_loads_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _extract_field(text: str, label: str) -> str:
    if not text:
        return ""
    pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)(?=\n[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÿ0-9 _/\-()]+:\s|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\n\s+", "\n", match.group(1).strip()).strip()


def _extract_numero(text: str) -> str:
    direct = _extract_field(text, "Número")
    if direct:
        return _upper(direct)
    match = re.search(r"\b(CHG\d{5,}|INC\d{5,})\b", text or "", re.IGNORECASE)
    return _upper(match.group(1)) if match else ""


def _extract_mes(text: str) -> str:
    direct = _extract_field(text, "Mês")
    target = direct or text or ""
    match = re.search(r"(20\d{2})[-/](\d{1,2})", target)
    if match:
        return f"{match.group(1)}-{match.group(2).zfill(2)}"
    return ""


def _extract_day(value: str) -> str:
    match = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", value or "")
    return match.group(3).zfill(2) if match else ""


def _sql_ident(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


class KnowledgeStructuredStore:
    TABLE = "knowledge_articles_structured"

    ANALYTIC_TERMS = {
        "quantos", "quantas", "quantidade", "qtd", "qtde", "total",
        "quais", "qual", "liste", "listar", "lista", "codigos", "códigos",
        "grupo de atribuicao", "grupo de atribuição", "ic impactado",
        "impactou", "referencia", "referência", "causado",
        "change", "changes", "chg", "mudanca", "mudança",
        "incidente", "incidentes", "inc",
        "app", "ecomm", "aura", "whatsapp", "dia", "mes", "mês",
    }

    BASE_COLUMNS: dict[str, str] = {
        "case_id": "TEXT",
        "knowledge_version": "TEXT",
        "agent_key": "TEXT",
        "agent_name": "TEXT",
        "project_key": "TEXT",
        "project_id": "TEXT",
        "project_name": "TEXT",
        "topic_id": "TEXT",
        "topic_ref_id": "TEXT",
        "topic_name": "TEXT",
        "topic_description": "TEXT",
        "article_id": "TEXT",
        "article_ref_id": "TEXT",
        "source_id": "TEXT",
        "codigo_tipo": "TEXT",
        "codigo_principal": "TEXT",
        "numero": "TEXT",
        "transacao_z": "TEXT",
        "mes": "TEXT",
        "tipo": "TEXT",
        "estado": "TEXT",
        "status": "TEXT",
        "grupo_atribuicao": "TEXT",
        "ic_impactado": "TEXT",
        "canal": "TEXT",
        "categoria": "TEXT",
        "prioridade": "TEXT",
        "data_inicio_planejada": "TEXT",
        "data_termino_planejada": "TEXT",
        "article_updated_on": "TEXT",
        "article_text": "TEXT",
        "raw_json": "TEXT",
        "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    }

    def __init__(self, db_path: str | None = None):
        default_path = (
            os.getenv("GABBI_NEXUS_DUCKDB_PATH")
            or os.getenv("DUCKDB_PATH")
            or "app/data/gabbi_nexus.duckdb"
        )
        self.db_path = Path(db_path or default_path)
        self.enabled = HAS_DUCKDB
        self.memory: dict[str, dict[str, Any]] = {}

        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("DuckDB não instalado")
        return duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        cols_sql = ",\n".join(
            f"{_sql_ident(col)} {definition}"
            for col, definition in self.BASE_COLUMNS.items()
        )
        with self._connect() as con:
            con.execute(f"CREATE TABLE IF NOT EXISTS {self.TABLE} ({cols_sql})")

    def _table_cols(self, con) -> list[str]:
        return [
            row[1]
            for row in con.execute(f"PRAGMA table_info('{self.TABLE}')").fetchall()
        ]

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "reason": "duckdb_not_installed"}

        try:
            with self._connect() as con:
                total = con.execute(f"SELECT COUNT(*) FROM {self.TABLE}").fetchone()[0]
                cases = con.execute(f"SELECT COUNT(DISTINCT case_id) FROM {self.TABLE}").fetchone()[0]
                cols = self._table_cols(con)
            return {
                "enabled": True,
                "db_path": str(self.db_path),
                "rows": int(total),
                "cases": int(cases),
                "columns": cols,
            }
        except Exception as exc:
            return {"enabled": False, "error": str(exc), "db_path": str(self.db_path)}

    def replace_case_rows(
        self,
        case_id: str,
        rows: list[dict[str, Any]],
        knowledge_version: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("DuckDB não está instalado. Instale com: pip install duckdb")

        normalized = [
            self._normalize_row(case_id, row, knowledge_version)
            for row in (rows or [])
        ]

        with self._connect() as con:
            table_cols = self._table_cols(con)
            table_col_set = set(table_cols)

            if "case_id" in table_col_set:
                con.execute(f"DELETE FROM {self.TABLE} WHERE case_id = ?", [case_id])
            else:
                con.execute(f"DELETE FROM {self.TABLE}")

            if normalized:
                insert_cols = [
                    col
                    for col in normalized[0].keys()
                    if col in table_col_set and col != "updated_at"
                ]

                if not insert_cols:
                    raise RuntimeError(
                        "Nenhuma coluna compatível encontrada. "
                        f"Tabela={table_cols}; Normalizado={list(normalized[0].keys())}"
                    )

                cols_sql = ", ".join(_sql_ident(c) for c in insert_cols)
                placeholders = ", ".join(["?"] * len(insert_cols))
                sql = f"INSERT INTO {self.TABLE} ({cols_sql}) VALUES ({placeholders})"

                values = [
                    [row.get(col) for col in insert_cols]
                    for row in normalized
                ]
                con.executemany(sql, values)

        self.memory.pop(case_id, None)

        return {
            "success": True,
            "case_id": case_id,
            "rows_received": len(rows or []),
            "rows_saved": len(normalized),
            "knowledge_version": knowledge_version,
        }

    def _normalize_row(
        self,
        case_id: str,
        row: dict[str, Any],
        knowledge_version: str | None,
    ) -> dict[str, Any]:
        raw_json = json.dumps(row, ensure_ascii=False, default=str)
        document_obj = _json_loads_maybe(row.get("document"))

        article_text = _safe(
            row.get("article")
            or row.get("article_text")
            or row.get("text")
            or ""
        )
        if not article_text:
            article_text = raw_json

        numero = (
            _upper(row.get("numero"))
            or _upper(row.get("Número"))
            or _upper(row.get("codigo_principal"))
            or _upper(document_obj.get("Número"))
            or _upper(document_obj.get("numero"))
            or _extract_numero(article_text)
        )

        if numero.startswith("CHG"):
            codigo_tipo = "CHG"
        elif numero.startswith("INC"):
            codigo_tipo = "INC"
        else:
            codigo_tipo = _upper(row.get("codigo_tipo")) or _upper(row.get("categoria")) or "OTHER"

        categoria = _upper(row.get("categoria")) or codigo_tipo

        mes = (
            _safe(row.get("mes"))
            or _safe(row.get("Mês"))
            or _safe(document_obj.get("Mês"))
            or _safe(document_obj.get("mes"))
            or _extract_mes(article_text)
        )
        mes = self._normalize_month(mes)

        data_inicio = (
            _safe(row.get("data_inicio_planejada"))
            or _safe(document_obj.get("Data de início planejada"))
            or _extract_field(article_text, "Data de início planejada")
        )

        data_termino = (
            _safe(row.get("data_termino_planejada"))
            or _safe(document_obj.get("Data de término planejada"))
            or _extract_field(article_text, "Data de término planejada")
        )

        aberto = (
            _safe(row.get("aberto"))
            or _safe(document_obj.get("Aberto"))
            or _extract_field(article_text, "Aberto")
        )

        if not mes:
            mes = self._normalize_month(data_inicio if codigo_tipo == "CHG" else aberto)

        dia = _extract_day(data_inicio if codigo_tipo == "CHG" else aberto)

        ic = (
            _safe(row.get("ic_impactado"))
            or _safe(document_obj.get("IC Impactado"))
            or _extract_field(article_text, "IC Impactado")
        )

        grupo = (
            _safe(row.get("grupo_atribuicao"))
            or _safe(row.get("grupo_de_atribuicao"))
            or _safe(document_obj.get("Grupo de atribuição"))
            or _extract_field(article_text, "Grupo de atribuição")
        )

        canal = (
            _safe(row.get("canal"))
            or _safe(row.get("canal_impactado"))
            or _safe(document_obj.get("Canal impactado"))
            or _safe(document_obj.get("Canal Impactado"))
            or _extract_field(article_text, "Canal impactado")
            or _extract_field(article_text, "Canal Impactado")
        )

        descricao_resumida = (
            _safe(row.get("descricao_resumida"))
            or _safe(document_obj.get("Descrição resumida"))
            or _extract_field(article_text, "Descrição resumida")
        )

        descricao = (
            _safe(row.get("descricao"))
            or _safe(document_obj.get("Descrição"))
            or _extract_field(article_text, "Descrição")
        )

        causado = (
            _safe(row.get("causado_pela_mudanca"))
            or _safe(document_obj.get("Causado pela mudança"))
            or _extract_field(article_text, "Causado pela mudança")
        )

        estado = (
            _safe(row.get("estado"))
            or _safe(document_obj.get("Estado"))
            or _extract_field(article_text, "Estado")
        )

        tipo = (
            _safe(row.get("tipo"))
            or _safe(document_obj.get("Tipo"))
            or _extract_field(article_text, "Tipo")
        )

        prioridade = (
            _safe(row.get("prioridade"))
            or _safe(document_obj.get("Prioridade"))
            or _extract_field(article_text, "Prioridade")
        )

        blob = _norm(" ".join([canal, descricao_resumida, descricao, article_text, raw_json]))

        is_app = bool(
            '"is_app": true' in raw_json.lower()
            or '"is_app":true' in raw_json.lower()
            or "canal impactado: app vivo" in blob
            or "canal impactado app vivo" in blob
            or "app vivo" in _norm(canal)
        )

        is_ecomm = bool(
            '"is_ecomm": true' in raw_json.lower()
            or '"is_ecomm":true' in raw_json.lower()
            or "ecomm" in blob
            or "e commerce" in blob
            or "ecommerce" in blob
            or "loja online" in blob
        )

        canal_enriched = canal
        if is_app and "APP_MARKER" not in canal_enriched:
            canal_enriched = (canal_enriched + " APP_MARKER").strip()
        if is_ecomm and "ECOMM_MARKER" not in canal_enriched:
            canal_enriched = (canal_enriched + " ECOMM_MARKER").strip()

        article_text_enriched = article_text
        if causado and "CAUSADO_PELA_MUDANCA_MARKER" not in article_text_enriched:
            article_text_enriched += f"\nCAUSADO_PELA_MUDANCA_MARKER: {causado}"
        if dia:
            article_text_enriched += f"\nDIA_NORMALIZADO_MARKER: {dia}"
        if is_app:
            article_text_enriched += "\nAPP_MARKER: true"
        if is_ecomm:
            article_text_enriched += "\nECOMM_MARKER: true"

        return {
            "case_id": case_id,
            "knowledge_version": knowledge_version,
            "agent_key": _safe(row.get("agent_key")),
            "agent_name": _safe(row.get("agent_name")),
            "project_key": _safe(row.get("project_key")),
            "project_id": _safe(row.get("project_id")),
            "project_name": _safe(row.get("project_name")),
            "topic_id": _safe(row.get("topic_id") or row.get("topicId")),
            "topic_ref_id": _safe(row.get("topic_ref_id")),
            "topic_name": _safe(row.get("topic_name")),
            "topic_description": _safe(row.get("topic_description")),
            "article_id": _safe(row.get("article_id") or row.get("id")),
            "article_ref_id": _safe(row.get("article_ref_id") or row.get("ref_id") or row.get("refId")),
            "source_id": _safe(row.get("source_id") or row.get("source")),
            "codigo_tipo": codigo_tipo,
            "codigo_principal": numero,
            "numero": numero,
            "transacao_z": _safe(row.get("transacao_z")),
            "mes": mes,
            "tipo": tipo,
            "estado": estado,
            "status": estado,
            "grupo_atribuicao": grupo,
            "ic_impactado": ic,
            "canal": canal_enriched,
            "categoria": categoria,
            "prioridade": prioridade,
            "data_inicio_planejada": data_inicio,
            "data_termino_planejada": data_termino,
            "article_updated_on": _safe(row.get("article_updated_on") or row.get("updated_on") or row.get("updatedOn")),
            "article_text": article_text_enriched,
            "raw_json": raw_json,
            "dia": dia,
            "is_app": is_app,
            "is_ecomm": is_ecomm,
            "descricao": descricao,
            "descricao_resumida": descricao_resumida,
            "causado_pela_mudanca": causado,
            "aberto": aberto,
        }

    def answer_question(
        self,
        case_id: str,
        question: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enabled:
            return {"fallback_to_rag": True}

        q = _norm(question)

        if not any(term in q for term in self.ANALYTIC_TERMS):
            return {"fallback_to_rag": True}

        try:
            return self._answer_structured(case_id, question)
        except Exception as exc:
            return {
                "fallback_to_rag": False,
                "route": "knowledge_structured_error",
                "query_type": "error",
                "answer_text": f"Erro ao consultar base estruturada: {exc}",
                "summary": f"Erro ao consultar base estruturada: {exc}",
                "technical": {"error": str(exc)},
            }

    def _answer_structured(self, case_id: str, question: str) -> dict[str, Any]:
        q = _norm(question)
        context = dict(self.memory.get(case_id) or {})

        if "quais os codigos" in q or "quais os códigos" in question.lower():
            codes = context.get("last_codes") or []
            return self._response(
                case_id,
                "\n".join(codes) if codes else "Não há códigos em memória para listar.",
                "list",
                context,
                {"memory_only": True, "codes": codes},
            )

        if "aura whatsapp" in q and context.get("last_codes"):
            return self._answer_aura_whatsapp(case_id, context)

        plan = self._build_plan(question, context)
        where_sql, params = self._build_where(case_id, plan)

        is_count = any(x in q for x in ["quantos", "quantas", "quantidade", "qtd", "qtde", "total"])
        is_list = any(x in q for x in ["quais", "qual", "liste", "listar", "lista", "mostre", "codigos", "códigos"])

        if plan.get("force_list"):
            is_list = True
            is_count = False

        if plan.get("force_count"):
            is_count = True
            is_list = False

        with self._connect() as con:
            distinct_expr = "COALESCE(NULLIF(numero, ''), NULLIF(codigo_principal, ''), article_id)"

            if is_count and not is_list:
                sql = f"""
                    SELECT COUNT(DISTINCT {distinct_expr}) AS total
                    FROM {self.TABLE}
                    {where_sql}
                """
                total = int(con.execute(sql, params).fetchone()[0] or 0)

                codes_sql = f"""
                    SELECT DISTINCT {distinct_expr} AS codigo
                    FROM {self.TABLE}
                    {where_sql}
                    ORDER BY codigo
                    LIMIT 2000
                """
                codes = [r[0] for r in con.execute(codes_sql, params).fetchall() if r[0]]
                self._save_memory(case_id, plan, codes)

                return self._response(
                    case_id,
                    str(total),
                    "count",
                    plan,
                    {"sql": sql, "params": params, "count": total, "codes_count": len(codes)},
                )

            if is_list:
                sql = f"""
                    SELECT DISTINCT {distinct_expr} AS codigo
                    FROM {self.TABLE}
                    {where_sql}
                    ORDER BY codigo
                    LIMIT 2000
                """
                codes = [r[0] for r in con.execute(sql, params).fetchall() if r[0]]
                self._save_memory(case_id, plan, codes)

                return self._response(
                    case_id,
                    "\n".join(codes) if codes else "Nenhum registro encontrado.",
                    "list",
                    plan,
                    {"sql": sql, "params": params, "count": len(codes)},
                )

        return self._response(
            case_id,
            "Não consegui calcular pela base estruturada.",
            "unknown",
            plan,
            {"reason": "unsupported_analytic_intent"},
        )

    def _build_plan(self, question: str, context: dict[str, Any]) -> dict[str, Any]:
        q = _norm(question)
        plan: dict[str, Any] = {}

        followup = (
            q.startswith("e ")
            or any(x in q for x in [
                "destes", "destas", "desses", "dessas", "deles", "delas",
                "essas", "esses", "foram abertas", "quais os codigos",
                "quais os códigos", "alguma dessas", "destes fazem",
            ])
        )

        if followup:
            plan.update(context.get("last_plan") or {})
            for key in ["codigo_tipo", "mes", "dia", "is_app", "is_ecomm"]:
                if context.get(key) is not None:
                    plan[key] = context[key]

        if any(x in q for x in ["change", "changes", "chg", "mudanca", "mudança"]):
            plan["codigo_tipo"] = "CHG"

        if any(x in q for x in ["incidente", "incidentes"]):
            plan["codigo_tipo"] = "INC"

        if ("referencia" in q or "referência" in q or "fazem referência" in q or "causado" in q) and re.search(r"\bchg\d{5,}\b", q):
            plan["codigo_tipo"] = "INC"

        month = self._extract_month_from_question(q)
        if month:
            plan["mes"] = month

        day = self._extract_day_from_question(q)
        if day:
            plan["dia"] = day

        ic = self._extract_after(question, r"ic\s+impactado\s+(.+)")
        if ic:
            plan["ic_impactado"] = ic
            plan["force_list"] = True

        grupo = self._extract_after(question, r"grupo\s+de\s+atribui[cç][aã]o\s*=?\s*(.+)")
        if grupo:
            plan["grupo_atribuicao"] = grupo
            plan["force_count"] = True

        chg_ref = re.search(r"\b(CHG\d{5,})\b", question, re.IGNORECASE)
        if chg_ref:
            plan["change_ref"] = chg_ref.group(1).upper()

        if re.search(r"\bapp\b", q):
            plan["is_app"] = True

        if any(x in q for x in ["ecomm", "e commerce", "e-commerce", "ecommerce"]):
            plan["is_ecomm"] = True

        return plan

    def _build_where(self, case_id: str, plan: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses = ["case_id = ?"]
        params: list[Any] = [case_id]

        codigo_tipo = plan.get("codigo_tipo")

        if codigo_tipo == "CHG":

            clauses.append(
                "numero LIKE 'CHG%'"
            )

        elif codigo_tipo == "INC":

            clauses.append(
                "numero LIKE 'INC%'"
            )

        if plan.get("mes"):
            clauses.append("mes = ?")
            params.append(plan["mes"])

        if plan.get("dia"):
            day_no_zero = str(int(plan["dia"]))
            day_zero = str(plan["dia"]).zfill(2)

            if codigo_tipo == "CHG":
                clauses.append(
                    """
                    (
                        regexp_extract(COALESCE(data_inicio_planejada, ''), '20[0-9]{2}[-/][0-9]{1,2}[-/]0?([0-9]{1,2})', 1) IN (?, ?)
                        OR article_text ILIKE ?
                    )
                    """
                )
                params.extend([day_no_zero, day_zero, f"%DIA_NORMALIZADO_MARKER: {day_zero}%"])

            elif codigo_tipo == "INC":
                clauses.append(
                    """
                    (
                        regexp_extract(COALESCE(article_text, ''), 'Aberto:\\s*20[0-9]{2}[-/][0-9]{1,2}[-/]0?([0-9]{1,2})', 1) IN (?, ?)
                        OR article_text ILIKE ?
                    )
                    """
                )
                params.extend([day_no_zero, day_zero, f"%DIA_NORMALIZADO_MARKER: {day_zero}%"])

        if plan.get("ic_impactado"):
            clauses.append("ic_impactado ILIKE ?")
            params.append(f"%{plan['ic_impactado']}%")

        if plan.get("grupo_atribuicao"):
            clauses.append("grupo_atribuicao ILIKE ?")
            params.append(f"%{plan['grupo_atribuicao']}%")

        if plan.get("is_app"):
            clauses.append(
                """
                (
                    canal ILIKE '%APP_MARKER%'
                    OR article_text ILIKE '%APP_MARKER: true%'
                    OR article_text ILIKE '%Canal Impactado: App Vivo%'
                    OR article_text ILIKE '%Canal impactado: APP Vivo%'
                    OR raw_json ILIKE '%"is_app": true%'
                    OR raw_json ILIKE '%"is_app":true%'
                )
                """
            )

        if plan.get("is_ecomm"):
            clauses.append(
                """
                (
                    canal ILIKE '%ECOMM_MARKER%'
                    OR article_text ILIKE '%ECOMM_MARKER: true%'
                    OR article_text ILIKE '%ecomm%'
                    OR article_text ILIKE '%e-commerce%'
                    OR article_text ILIKE '%ecommerce%'
                    OR raw_json ILIKE '%"is_ecomm": true%'
                    OR raw_json ILIKE '%"is_ecomm":true%'
                )
                """
            )

        if plan.get("change_ref"):
            clauses.append(
                """
                (
                    article_text ILIKE ?
                    OR raw_json ILIKE ?
                )
                """
            )
            ref = f"%{plan['change_ref']}%"
            params.extend([ref, ref])

        return "WHERE " + " AND ".join(clauses), params

    def _answer_aura_whatsapp(self, case_id: str, context: dict[str, Any]) -> dict[str, Any]:
        codes = context.get("last_codes") or []
        if not codes:
            return self._response(case_id, "Não", "boolean", context, {"reason": "no_last_codes"})

        placeholders = ",".join(["?"] * len(codes))
        params = [case_id] + list(codes)

        sql = f"""
            SELECT DISTINCT numero
            FROM {self.TABLE}
            WHERE case_id = ?
            AND numero IN ({placeholders})
            AND (
                ic_impactado ILIKE '%AURA WHATSAPP%'
                OR article_text ILIKE '%AURA WHATSAPP%'
                OR raw_json ILIKE '%AURA WHATSAPP%'
            )
            ORDER BY numero
        """

        with self._connect() as con:
            found = [r[0] for r in con.execute(sql, params).fetchall() if r[0]]

        return self._response(
            case_id,
            "Sim: " + ", ".join(found) if found else "Não",
            "boolean",
            context,
            {"sql": sql, "found": found, "codes": codes},
        )

    def _save_memory(self, case_id: str, plan: dict[str, Any], codes: list[str]) -> None:
        memory = dict(self.memory.get(case_id) or {})
        memory["last_plan"] = dict(plan)
        for key in ["codigo_tipo", "mes", "dia", "is_app", "is_ecomm"]:
            if key in plan:
                memory[key] = plan[key]
        memory["last_codes"] = codes
        self.memory[case_id] = memory

    def _response(
        self,
        case_id: str,
        answer: str,
        query_type: str,
        criteria: dict[str, Any],
        technical: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "fallback_to_rag": False,
            "route": "knowledge_structured_duckdb",
            "query_type": query_type,
            "answer_text": answer,
            "summary": answer,
            "technical": {
                "case_id": case_id,
                "criteria": criteria,
                **(technical or {}),
            },
            "sources": {
                "deterministic": True,
                "engine": "duckdb",
                "table": self.TABLE,
            },
        }

    def _normalize_month(self, value: Any) -> str:
        match = re.search(r"(20\d{2})[-/](\d{1,2})", _safe(value))
        return f"{match.group(1)}-{match.group(2).zfill(2)}" if match else ""

    def _extract_month_from_question(self, q: str) -> str:
        patterns = [
            r"m[eê]s\s+(\d{1,2})\s+de\s+(20\d{2})",
            r"\b(\d{1,2})\s+de\s+(20\d{2})\b",
            r"\b(20\d{2})[-/](\d{1,2})\b",
        ]

        for pattern in patterns:
            match = re.search(pattern, q)
            if not match:
                continue

            if pattern.startswith(r"\b(20"):
                yyyy = match.group(1)
                mm = match.group(2).zfill(2)
            else:
                mm = match.group(1).zfill(2)
                yyyy = match.group(2)

            if 1 <= int(mm) <= 12:
                return f"{yyyy}-{mm}"

        return ""

    def _extract_day_from_question(self, q: str) -> str:
        match = re.search(r"\bdia\s+(\d{1,2})\b", q)
        if not match:
            return ""

        day = match.group(1).zfill(2)
        return day if 1 <= int(day) <= 31 else ""

    def _extract_after(self, question: str, pattern: str) -> str:
        match = re.search(pattern, question, re.IGNORECASE)
        if not match:
            return ""

        value = match.group(1).strip().replace("?", "").strip()
        lower = value.lower()
        cut = len(value)

        for stop in ["\n", " e grupo ", " e com ", " e que ", " e quais ", " e quant"]:
            idx = lower.find(stop)
            if idx >= 0:
                cut = min(cut, idx)

        return value[:cut].strip()