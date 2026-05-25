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
    if value is None:
        return ""
    return str(value).strip()


def _norm(value: Any) -> str:
    text = _safe(value)
    text = unicodedata.normalize("NFKD", text)
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


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value

    text = _norm(value)

    if text in {"true", "1", "sim", "yes"}:
        return True

    if text in {"false", "0", "nao", "não", "no"}:
        return False

    return None


def _extract_field(text: str, label: str) -> str:
    if not text:
        return ""

    pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)(?=\n[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÿ0-9 _/\-()]+:\s|\Z)"

    match = re.search(pattern, text, flags=re.DOTALL)

    if not match:
        return ""

    value = match.group(1).strip()

    return re.sub(r"\n\s+", "\n", value).strip()


def _extract_numero(text: str) -> str:
    direct = _extract_field(text, "Número")

    if direct:
        return _upper(direct)

    match = re.search(
        r"\b(CHG\d{5,}|INC\d{5,})\b",
        text or "",
        re.IGNORECASE,
    )

    if match:
        return _upper(match.group(1))

    return ""


def _extract_mes(text: str) -> str:
    direct = _extract_field(text, "Mês")

    if direct:
        m = re.search(r"(20\d{2})[-/](\d{1,2})", direct)

        if m:
            return f"{m.group(1)}-{m.group(2).zfill(2)}"

    m = re.search(
        r"(20\d{2})[-/](\d{1,2})[-/]\d{1,2}",
        text or "",
    )

    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}"

    return ""


def _extract_day(value: str) -> str:
    m = re.search(
        r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})",
        value or "",
    )

    if not m:
        return ""

    return m.group(3).zfill(2)


class KnowledgeStructuredStore:

    TABLE = "knowledge_articles_structured"

    ANALYTIC_TERMS = {
        "quantos",
        "quantas",
        "quantidade",
        "qtd",
        "qtde",
        "total",
        "quais",
        "qual",
        "liste",
        "listar",
        "lista",
        "codigos",
        "códigos",
        "grupo de atribuicao",
        "grupo de atribuição",
        "ic impactado",
        "impactou",
        "referencia",
        "referência",
        "causado",
        "change",
        "changes",
        "chg",
        "incidente",
        "incidentes",
        "inc",
        "app",
        "ecomm",
        "aura",
        "whatsapp",
        "dia",
        "mes",
        "mês",
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

    def _connect(self):

        if not self.enabled:
            raise RuntimeError(
                "DuckDB não instalado"
            )

        return duckdb.connect(str(self.db_path))

    def replace_case_rows(
        self,
        case_id: str,
        rows: list[dict[str, Any]],
        knowledge_version: str | None = None,
    ) -> dict[str, Any]:

        if not self.enabled:
            raise RuntimeError(
                "DuckDB não está instalado. "
                "Instale com: pip install duckdb"
            )

        if not case_id:
            raise ValueError("case_id é obrigatório")

        normalized = [
            self._normalize_row(
                case_id,
                row,
                knowledge_version,
            )
            for row in (rows or [])
        ]

        with self._connect() as con:

            table_cols = [
                row[1]
                for row in con.execute(
                    f"PRAGMA table_info('{self.TABLE}')"
                ).fetchall()
            ]

            table_col_set = set(table_cols)

            if "case_id" in table_col_set:
                con.execute(
                    f"""
                    DELETE FROM {self.TABLE}
                    WHERE case_id = ?
                    """,
                    [case_id],
                )
            else:
                con.execute(
                    f"DELETE FROM {self.TABLE}"
                )

            if normalized:

                insert_cols = [
                    col
                    for col in normalized[0].keys()
                    if col in table_col_set
                ]

                if not insert_cols:
                    raise RuntimeError(
                        "Nenhuma coluna compatível encontrada.\n"
                        f"Tabela={table_cols}\n"
                        f"Normalizado={list(normalized[0].keys())}"
                    )

                cols_sql = ", ".join(
                    f'"{c}"'
                    for c in insert_cols
                )

                placeholders = ", ".join(
                    ["?"] * len(insert_cols)
                )

                sql = f"""
                    INSERT INTO {self.TABLE}
                    ({cols_sql})
                    VALUES ({placeholders})
                """

                values = [
                    [
                        row.get(col)
                        for col in insert_cols
                    ]
                    for row in normalized
                ]

                con.executemany(
                    sql,
                    values,
                )

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

        raw_json = json.dumps(
            row,
            ensure_ascii=False,
            default=str,
        )

        document_obj = _json_loads_maybe(
            row.get("document")
        )

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
            or _upper(document_obj.get("Número"))
            or _upper(document_obj.get("numero"))
            or _extract_numero(article_text)
        )

        categoria = (
            _upper(row.get("categoria"))
            or _upper(document_obj.get("categoria"))
            or ""
        )

        codigo_tipo = (
            _upper(row.get("codigo_tipo"))
            or categoria
        )

        if numero.startswith("CHG"):
            codigo_tipo = "CHG"

        elif numero.startswith("INC"):
            codigo_tipo = "INC"

        elif not codigo_tipo:
            codigo_tipo = "OTHER"

        mes = (
            _safe(row.get("mes"))
            or _safe(document_obj.get("mes"))
            or _extract_mes(article_text)
        )

        m = re.search(
            r"(20\d{2})[-/](\d{1,2})",
            mes,
        )

        if m:
            mes = f"{m.group(1)}-{m.group(2).zfill(2)}"
        else:
            mes = ""

        data_inicio = (
            _safe(row.get("data_inicio_planejada"))
            or _safe(document_obj.get("Data de início planejada"))
            or _extract_field(
                article_text,
                "Data de início planejada",
            )
        )

        aberto = (
            _safe(row.get("aberto"))
            or _safe(document_obj.get("Aberto"))
            or _extract_field(
                article_text,
                "Aberto",
            )
        )

        if codigo_tipo == "CHG":
            dia = _extract_day(data_inicio)

        elif codigo_tipo == "INC":
            dia = _extract_day(aberto)

        else:
            dia = _extract_day(
                data_inicio or aberto
            )

        ic = (
            _safe(row.get("ic_impactado"))
            or _safe(document_obj.get("IC Impactado"))
            or _extract_field(
                article_text,
                "IC Impactado",
            )
        )

        grupo = (
            _safe(row.get("grupo_atribuicao"))
            or _safe(document_obj.get("Grupo de atribuição"))
            or _extract_field(
                article_text,
                "Grupo de atribuição",
            )
        )

        descricao = (
            _safe(row.get("descricao"))
            or _extract_field(
                article_text,
                "Descrição resumida",
            )
        )

        raw_is_app = (
            row.get("is_app")
            if row.get("is_app") is not None
            else document_obj.get("is_app")
        )

        raw_is_ecomm = (
            row.get("is_ecomm")
            if row.get("is_ecomm") is not None
            else document_obj.get("is_ecomm")
        )

        is_app = _bool_value(raw_is_app)
        is_ecomm = _bool_value(raw_is_ecomm)

        search_blob = _norm(
            " ".join(
                [
                    grupo,
                    ic,
                    descricao,
                    article_text,
                ]
            )
        )

        if is_app is None:
            is_app = bool(
                re.search(r"\bapp\b", search_blob)
            )

        if is_ecomm is None:
            is_ecomm = bool(
                "ecomm" in search_blob
                or "ecommerce" in search_blob
                or "e-commerce" in search_blob
            )

        return {
            "case_id": case_id,
            "knowledge_version": knowledge_version,
            "codigo_tipo": codigo_tipo,
            "numero": numero,
            "mes": mes,
            "dia": dia,
            "categoria": categoria,
            "grupo_atribuicao": grupo,
            "ic_impactado": ic,
            "descricao": descricao,
            "article_text": article_text,
            "raw_json": raw_json,
            "is_app": bool(is_app),
            "is_ecomm": bool(is_ecomm),
        }

    def answer_question(
        self,
        case_id: str,
        question: str,
        chat_history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:

        if not self.enabled:
            return {
                "fallback_to_rag": True
            }

        q = _norm(question)

        if not any(
            term in q
            for term in self.ANALYTIC_TERMS
        ):
            return {
                "fallback_to_rag": True
            }

        try:
            return self._answer_structured(
                case_id,
                question,
            )

        except Exception as exc:

            return {
                "fallback_to_rag": False,
                "route": "knowledge_structured_error",
                "query_type": "error",
                "answer_text": str(exc),
                "technical": {
                    "error": str(exc),
                },
            }

    def _answer_structured(
        self,
        case_id: str,
        question: str,
    ) -> dict[str, Any]:

        q = _norm(question)

        context = dict(
            self.memory.get(case_id) or {}
        )

        if (
            "quais os codigos" in q
            or "quais os códigos" in q
        ):

            codes = context.get("last_codes") or []

            return {
                "fallback_to_rag": False,
                "route": "knowledge_structured_memory",
                "query_type": "list",
                "answer_text": "\n".join(codes),
            }

        if "aura whatsapp" in q and context.get("last_codes"):

            codes = context.get("last_codes") or []

            placeholders = ",".join(
                ["?"] * len(codes)
            )

            params = [case_id] + codes

            sql = f"""
                SELECT DISTINCT numero
                FROM {self.TABLE}
                WHERE case_id = ?
                AND numero IN ({placeholders})
                AND (
                    ic_impactado ILIKE '%AURA WHATSAPP%'
                    OR article_text ILIKE '%AURA WHATSAPP%'
                )
                ORDER BY numero
            """

            with self._connect() as con:

                found = [
                    r[0]
                    for r in con.execute(
                        sql,
                        params,
                    ).fetchall()
                ]

            return {
                "fallback_to_rag": False,
                "route": "knowledge_structured_aura",
                "query_type": "boolean",
                "answer_text": (
                    "Sim: " + ", ".join(found)
                    if found
                    else "Não"
                ),
            }

        plan = self._build_plan(
            question,
            context,
        )

        where_sql, params = self._build_where(
            case_id,
            plan,
        )

        is_count = any(
            x in q
            for x in [
                "quantos",
                "quantas",
                "quantidade",
                "total",
            ]
        )

        is_list = any(
            x in q
            for x in [
                "quais",
                "liste",
                "listar",
                "lista",
            ]
        )

        if plan.get("force_count"):
            is_count = True
            is_list = False

        with self._connect() as con:

            distinct_expr = """
            COALESCE(
                NULLIF(numero, ''),
                'SEM_NUMERO'
            )
            """

            if is_count and not is_list:

                sql = f"""
                    SELECT COUNT(
                        DISTINCT {distinct_expr}
                    )
                    FROM {self.TABLE}
                    {where_sql}
                """

                total = int(
                    con.execute(
                        sql,
                        params,
                    ).fetchone()[0]
                    or 0
                )

                codes_sql = f"""
                    SELECT DISTINCT
                    {distinct_expr} AS codigo
                    FROM {self.TABLE}
                    {where_sql}
                    ORDER BY codigo
                """

                codes = [
                    r[0]
                    for r in con.execute(
                        codes_sql,
                        params,
                    ).fetchall()
                ]

                self._save_memory(
                    case_id,
                    plan,
                    codes,
                )

                return {
                    "fallback_to_rag": False,
                    "route": "knowledge_structured_count",
                    "query_type": "count",
                    "answer_text": str(total),
                    "technical": {
                        "sql": sql,
                        "params": params,
                    },
                }

            if is_list:

                sql = f"""
                    SELECT DISTINCT
                    {distinct_expr} AS codigo
                    FROM {self.TABLE}
                    {where_sql}
                    ORDER BY codigo
                """

                codes = [
                    r[0]
                    for r in con.execute(
                        sql,
                        params,
                    ).fetchall()
                ]

                self._save_memory(
                    case_id,
                    plan,
                    codes,
                )

                return {
                    "fallback_to_rag": False,
                    "route": "knowledge_structured_list",
                    "query_type": "list",
                    "answer_text": "\n".join(codes),
                    "technical": {
                        "sql": sql,
                        "params": params,
                    },
                }

        return {
            "fallback_to_rag": False,
            "route": "knowledge_structured_no_answer",
            "answer_text": (
                "Não consegui calcular "
                "pela base estruturada."
            ),
        }

    def _build_plan(
        self,
        question: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:

        q = _norm(question)

        plan: dict[str, Any] = {}

        followup = (
            q.startswith("e ")
            or "destes" in q
            or "destas" in q
            or "desses" in q
            or "dessas" in q
            or "deles" in q
            or "delas" in q
            or "essas" in q
            or "esses" in q
            or "foram abertas" in q
            or "quais os codigos" in q
            or "quais os códigos" in q
            or "alguma dessas" in q
        )

        if followup:

            for key in [
                "codigo_tipo",
                "mes",
                "dia",
                "is_app",
                "is_ecomm",
            ]:
                if context.get(key) is not None:
                    plan[key] = context[key]

        if any(
            x in q
            for x in [
                "change",
                "changes",
                "chg",
            ]
        ):
            plan["codigo_tipo"] = "CHG"

        if (
            (
                "referencia" in q
                or "referência" in q
                or "fazem referência" in q
            )
            and re.search(r"\bchg\d{5,}\b", q)
        ):
            plan["codigo_tipo"] = "INC"

        if any(
            x in q
            for x in [
                "incidente",
                "incidentes",
            ]
        ):
            plan["codigo_tipo"] = "INC"

        month = self._extract_month_from_question(q)

        if month:
            plan["mes"] = month

        day = self._extract_day_from_question(q)

        if day:
            plan["dia"] = day

        ic = self._extract_after(
            question,
            r"ic\s+impactado\s+(.+)",
        )

        if ic:
            plan["ic_impactado"] = ic

        grupo = self._extract_after(
            question,
            r"grupo\s+de\s+atribui[cç][aã]o\s*=?\s*(.+)",
        )

        if grupo:
            plan["grupo_atribuicao"] = grupo
            plan["force_count"] = True

        if re.search(r"\bapp\b", q):
            plan["is_app"] = True

        if any(
            x in q
            for x in [
                "ecomm",
                "ecommerce",
                "e-commerce",
            ]
        ):
            plan["is_ecomm"] = True

        return plan

    def _build_where(
        self,
        case_id: str,
        plan: dict[str, Any],
    ) -> tuple[str, list[Any]]:

        clauses = [
            "case_id = ?"
        ]

        params: list[Any] = [
            case_id
        ]

        codigo_tipo = plan.get("codigo_tipo")

        if codigo_tipo:

            clauses.append(
                "UPPER(codigo_tipo)=UPPER(?)"
            )

            params.append(codigo_tipo)

            if codigo_tipo == "CHG":

                clauses.append(
                    "numero LIKE 'CHG%'"
                )

            elif codigo_tipo == "INC":

                clauses.append(
                    "numero LIKE 'INC%'"
                )

        if plan.get("mes"):

            clauses.append(
                "mes = ?"
            )

            params.append(
                plan["mes"]
            )

        if plan.get("dia"):

            day_no_zero = str(
                int(plan["dia"])
            )

            if plan.get("codigo_tipo") == "CHG":

                clauses.append(
                    """
                    regexp_extract(
                        COALESCE(article_text, ''),
                        'Data de início planejada:\\s*20[0-9]{2}[-/][0-9]{1,2}[-/]0?([0-9]{1,2})',
                        1
                    ) = ?
                    """
                )

                params.append(day_no_zero)

            elif plan.get("codigo_tipo") == "INC":

                clauses.append(
                    """
                    regexp_extract(
                        COALESCE(article_text, ''),
                        'Aberto:\\s*20[0-9]{2}[-/][0-9]{1,2}[-/]0?([0-9]{1,2})',
                        1
                    ) = ?
                    """
                )

                params.append(day_no_zero)

        if plan.get("ic_impactado"):

            clauses.append(
                "ic_impactado ILIKE ?"
            )

            params.append(
                f"%{plan['ic_impactado']}%"
            )

        if plan.get("grupo_atribuicao"):

            clauses.append(
                "grupo_atribuicao ILIKE ?"
            )

            params.append(
                f"%{plan['grupo_atribuicao']}%"
            )

        if plan.get("is_app"):

            clauses.append(
                """
                (
                    article_text ILIKE '%Canal Impactado: App Vivo%'
                    OR article_text ILIKE '%Canal impactado: APP Vivo%'
                    OR article_text ILIKE '%Canal impactado: App Vivo%'
                    OR raw_json ILIKE '%"is_app": true%'
                    OR raw_json ILIKE '%"is_app":true%'
                )
                """
            )

        if plan.get("is_ecomm"):

            clauses.append(
                """
                (
                    article_text ILIKE '%ecomm%'
                    OR article_text ILIKE '%e-commerce%'
                    OR article_text ILIKE '%ecommerce%'
                )
                """
            )

        return (
            "WHERE " + " AND ".join(clauses),
            params,
        )

    def _save_memory(
        self,
        case_id: str,
        plan: dict[str, Any],
        codes: list[str],
    ) -> None:

        memory = dict(
            self.memory.get(case_id)
            or {}
        )

        for key in [
            "codigo_tipo",
            "mes",
            "dia",
            "is_app",
            "is_ecomm",
        ]:
            if key in plan:
                memory[key] = plan[key]

        memory["last_codes"] = codes

        self.memory[case_id] = memory

    def _extract_month_from_question(
        self,
        q: str,
    ) -> str:

        patterns = [
            r"m[eê]s\s+(\d{1,2})\s+de\s+(20\d{2})",
            r"\b(\d{1,2})\s+de\s+(20\d{2})\b",
            r"\b(20\d{2})[-/](\d{1,2})\b",
        ]

        for pattern in patterns:

            m = re.search(
                pattern,
                q,
            )

            if not m:
                continue

            if pattern.startswith(r"\b(20"):

                yyyy = m.group(1)
                mm = m.group(2).zfill(2)

            else:

                mm = m.group(1).zfill(2)
                yyyy = m.group(2)

            if 1 <= int(mm) <= 12:
                return f"{yyyy}-{mm}"

        return ""

    def _extract_day_from_question(
        self,
        q: str,
    ) -> str:

        m = re.search(
            r"\bdia\s+(\d{1,2})\b",
            q,
        )

        if not m:
            return ""

        day = m.group(1).zfill(2)

        if 1 <= int(day) <= 31:
            return day

        return ""

    def _extract_after(
        self,
        question: str,
        pattern: str,
    ) -> str:

        m = re.search(
            pattern,
            question,
            re.IGNORECASE,
        )

        if not m:
            return ""

        value = (
            m.group(1)
            .strip()
            .replace("?", "")
            .strip()
        )

        lower = value.lower()

        cut = len(value)

        for stop in [
            "\n",
            " e grupo ",
            " e com ",
            " e que ",
        ]:

            idx = lower.find(stop)

            if idx >= 0:
                cut = min(cut, idx)

        return value[:cut].strip()