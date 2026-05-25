from __future__ import annotations

import os
import re
import json
import duckdb
import pandas as pd
from typing import Any


class KnowledgeStructuredStore:

    ANALYTIC_TERMS = {
        "quantos",
        "quantas",
        "quais",
        "liste",
        "listar",
        "lista",
        "mostre",
        "contagem",
        "count",
        "grupo",
        "agrupamento",
        "ic impactado",
        "grupo de atribuição",
        "grupo de atribuicao",
        "mudança",
        "mudanca",
        "change",
        "incidente",
        "incident",
    }

    def __init__(self):
        self.base_dir = os.getenv("KNOWLEDGE_STRUCTURED_DIR", "data/structured")
        os.makedirs(self.base_dir, exist_ok=True)

        self.memory: dict[str, dict[str, Any]] = {}

    # ============================================================
    # STATUS
    # ============================================================

    def status(self):
        return {
            "enabled": True,
            "base_dir": self.base_dir,
        }

    # ============================================================
    # STORAGE
    # ============================================================

    def _db_path(self, case_id: str):
        return os.path.join(self.base_dir, f"{case_id}.duckdb")

    def _connect(self, case_id: str):
        return duckdb.connect(self._db_path(case_id))

    # ============================================================
    # UPSERT
    # ============================================================

    def replace_case_rows(
        self,
        case_id: str,
        rows: list[dict],
        knowledge_version: str | None = None,
    ):

        conn = self._connect(case_id)

        normalized = []

        for row in rows:

            row = self._normalize_row(row)

            numero = str(row.get("numero", "")).upper().strip()

            # IGNORA registros sem número
            if not numero:
                continue

            normalized.append(row)

        df = pd.DataFrame(normalized)

        conn.execute("DROP TABLE IF EXISTS knowledge")

        conn.register("df_temp", df)

        conn.execute("""
            CREATE TABLE knowledge AS
            SELECT *
            FROM df_temp
        """)

        return {
            "success": True,
            "case_id": case_id,
            "rows_received": len(rows),
            "rows_saved": len(df),
            "knowledge_version": knowledge_version,
        }

    # ============================================================
    # NORMALIZAÇÃO
    # ============================================================

    def _normalize_row(self, row):

        normalized = {}

        for k, v in row.items():

            key = (
                str(k)
                .strip()
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("/", "_")
            )

            value = "" if v is None else str(v).strip()

            normalized[key] = value

        numero = normalized.get("numero", "").upper()

        # tipo lógico
        if numero.startswith("CHG"):
            normalized["codigo_tipo"] = "CHG"

        elif numero.startswith("INC"):
            normalized["codigo_tipo"] = "INC"

        else:
            normalized["codigo_tipo"] = "OTHER"

        # mes yyyy-mm
        data_ref = (
            normalized.get("data_inicio_planejada")
            or normalized.get("aberto_em")
            or normalized.get("created_on")
            or ""
        )

        mes_match = re.search(r"(20\d{2})[-/](\d{2})", data_ref)

        if mes_match:
            normalized["mes"] = f"{mes_match.group(1)}-{mes_match.group(2)}"

        # dia
        dia_match = re.search(r"(20\d{2})[-/](\d{2})[-/](\d{2})", data_ref)

        if dia_match:
            normalized["dia"] = dia_match.group(3)

        # app/ecomm
        text_blob = json.dumps(normalized, ensure_ascii=False).lower()

        normalized["is_app"] = "app" in text_blob
        normalized["is_ecomm"] = (
            "ecomm" in text_blob
            or "e-commerce" in text_blob
            or "ecommerce" in text_blob
        )

        return normalized

    # ============================================================
    # MAIN ASK
    # ============================================================

    def answer_question(
        self,
        case_id: str,
        question: str,
        chat_history: list[dict] | None = None,
    ):

        q = question.lower().strip()

        if not self._is_analytic(q):
            return {
                "fallback_to_rag": True
            }

        conn = self._connect(case_id)

        # ========================================================
        # FOLLOW-UP MEMORY
        # ========================================================

        context = self.memory.get(case_id, {})

        filters = {
            "codigo_tipo": context.get("codigo_tipo"),
            "mes": context.get("mes"),
            "dia": context.get("dia"),
        }

        # ========================================================
        # MÊS
        # ========================================================

        mes_match = re.search(
            r"(?:m[eê]s)\s+(\d{1,2})\s+(?:de)\s+(20\d{2})",
            q
        )

        if mes_match:
            mm = mes_match.group(1).zfill(2)
            yyyy = mes_match.group(2)
            filters["mes"] = f"{yyyy}-{mm}"

        # ========================================================
        # DIA
        # ========================================================

        dia_match = re.search(
            r"(?:dia)\s+(\d{1,2})",
            q
        )

        if dia_match:
            filters["dia"] = dia_match.group(1).zfill(2)

        # ========================================================
        # CHG / INC
        # ========================================================

        if "change" in q or "changes" in q or "chg" in q:
            filters["codigo_tipo"] = "CHG"

        if "incidente" in q or "incidentes" in q or "inc" in q:
            filters["codigo_tipo"] = "INC"

        # ========================================================
        # IC IMPACTADO
        # ========================================================

        ic_match = re.search(
            r"ic impactado\s+([a-zA-Z0-9_\- ]+)",
            q,
            re.IGNORECASE
        )

        ic_value = None

        if ic_match:
            ic_value = ic_match.group(1).strip()

        # ========================================================
        # GRUPO
        # ========================================================

        grupo_match = re.search(
            r"grupo de atribui[cç][aã]o\s*=?\s*([a-zA-Z0-9_\-]+)",
            q,
            re.IGNORECASE
        )

        grupo_value = None

        if grupo_match:
            grupo_value = grupo_match.group(1).strip()

        # ========================================================
        # CHG REFERENCIADA
        # ========================================================

        change_ref_match = re.search(
            r"(CHG\d+)",
            q,
            re.IGNORECASE
        )

        change_ref = None

        if change_ref_match:
            change_ref = change_ref_match.group(1).upper()

        # ========================================================
        # SQL
        # ========================================================

        where = []

        if filters.get("codigo_tipo"):
            where.append(
                f"codigo_tipo = '{filters['codigo_tipo']}'"
            )

        if filters.get("mes"):
            where.append(
                f"mes = '{filters['mes']}'"
            )

        if filters.get("dia"):
            where.append(
                f"dia = '{filters['dia']}'"
            )

        # CRÍTICO:
        # nunca considerar CHG referenciada em INC
        if filters.get("codigo_tipo") == "CHG":
            where.append(
                "numero LIKE 'CHG%'"
            )

        if filters.get("codigo_tipo") == "INC":
            where.append(
                "numero LIKE 'INC%'"
            )

        if ic_value:
            where.append(
                f"LOWER(ic_impactado) LIKE LOWER('%{ic_value}%')"
            )

        if grupo_value:
            where.append(
                f"LOWER(grupo_de_atribuicao) LIKE LOWER('%{grupo_value}%')"
            )

        if "app" in q:
            where.append("is_app = true")

        if "ecomm" in q or "e-commerce" in q:
            where.append("is_ecomm = true")

        if change_ref:
            where.append(
                f"""
                (
                    LOWER(causado_pela_mudanca) LIKE LOWER('%{change_ref}%')
                    OR LOWER(change_relacionada) LIKE LOWER('%{change_ref}%')
                    OR LOWER(descricao) LIKE LOWER('%{change_ref}%')
                )
                """
            )

        where_sql = " AND ".join(where)

        if where_sql:
            where_sql = "WHERE " + where_sql

        # ========================================================
        # COUNT
        # ========================================================

        is_count = any(
            t in q
            for t in [
                "quantos",
                "quantas",
                "quantidade",
                "count",
            ]
        )

        # ========================================================
        # LIST
        # ========================================================

        is_list = any(
            t in q
            for t in [
                "quais",
                "liste",
                "listar",
                "mostre",
                "códigos",
                "codigos",
            ]
        )

        # ========================================================
        # COUNT DISTINCT
        # ========================================================

        if is_count and not is_list:

            sql = f"""
                SELECT COUNT(DISTINCT numero) AS total
                FROM knowledge
                {where_sql}
            """

            result = conn.execute(sql).fetchone()

            total = int(result[0] or 0)

            self.memory[case_id] = filters

            return {
                "fallback_to_rag": False,
                "route": "structured_analytics",
                "query_type": "count",
                "answer_text": str(total),
                "technical": {
                    "sql": sql,
                    "filters": filters,
                }
            }

        # ========================================================
        # LIST DISTINCT
        # ========================================================

        if is_list:

            sql = f"""
                SELECT DISTINCT numero
                FROM knowledge
                {where_sql}
                ORDER BY numero
            """

            rows = conn.execute(sql).fetchall()

            values = [r[0] for r in rows]

            self.memory[case_id] = filters
            self.memory[case_id]["last_codes"] = values

            return {
                "fallback_to_rag": False,
                "route": "structured_analytics",
                "query_type": "list",
                "answer_text": "\n".join(values),
                "technical": {
                    "sql": sql,
                    "filters": filters,
                    "count": len(values),
                }
            }

        return {
            "fallback_to_rag": True
        }

    # ============================================================
    # ANALYTIC DETECTION
    # ============================================================

    def _is_analytic(self, q: str):

        for term in self.ANALYTIC_TERMS:
            if term in q:
                return True

        return False