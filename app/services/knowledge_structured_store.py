from __future__ import annotations

import os
import re
import duckdb
import pandas as pd
from typing import Any


class KnowledgeStructuredStore:

    def __init__(self):

        self.base_dir = os.getenv(
            "KNOWLEDGE_STRUCTURED_DIR",
            "data/structured"
        )

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
    # DB
    # ============================================================

    def _db_path(self, case_id: str):

        return os.path.join(
            self.base_dir,
            f"{case_id}.duckdb"
        )

    def _connect(self, case_id: str):

        return duckdb.connect(
            self._db_path(case_id)
        )

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

            r = self._normalize_row(row)

            numero = str(
                r.get("numero", "")
            ).upper().strip()

            if not numero:
                continue

            normalized.append(r)

        df = pd.DataFrame(normalized)

        conn.execute(
            "DROP TABLE IF EXISTS knowledge"
        )

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
    # NORMALIZE
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

            value = (
                "" if v is None else str(v).strip()
            )

            normalized[key] = value

        numero = normalized.get(
            "numero",
            ""
        ).upper()

        # ========================================================
        # ENTITY TYPE
        # ========================================================

        if numero.startswith("CHG"):
            normalized["codigo_tipo"] = "CHG"

        elif numero.startswith("INC"):
            normalized["codigo_tipo"] = "INC"

        else:
            normalized["codigo_tipo"] = "OTHER"

        # ========================================================
        # MES
        # ========================================================

        data_ref = (
            normalized.get("data_inicio_planejada")
            or normalized.get("aberto_em")
            or normalized.get("created_on")
            or normalized.get("inicio")
            or ""
        )

        mes_match = re.search(
            r"(20\d{2})[-/](\d{2})",
            data_ref
        )

        if mes_match:

            normalized["mes"] = (
                f"{mes_match.group(1)}-"
                f"{mes_match.group(2)}"
            )

        # ========================================================
        # DIA
        # ========================================================

        dia_match = re.search(
            r"(20\d{2})[-/](\d{2})[-/](\d{2})",
            data_ref
        )

        if dia_match:
            normalized["dia"] = dia_match.group(3)

        # ========================================================
        # APP / ECOMM
        # ========================================================

        blob = " ".join(
            str(v).lower()
            for v in normalized.values()
        )

        normalized["is_app"] = (
            " app " in f" {blob} "
            or "app_" in blob
            or "_app" in blob
        )

        normalized["is_ecomm"] = (
            "ecomm" in blob
            or "e-commerce" in blob
            or "ecommerce" in blob
        )

        return normalized

    # ============================================================
    # ASK
    # ============================================================

    def answer_question(
        self,
        case_id: str,
        question: str,
        chat_history: list[dict] | None = None,
    ):

        q = question.lower().strip()

        conn = self._connect(case_id)

        context = self.memory.get(case_id, {})

        filters = {
            "codigo_tipo": context.get("codigo_tipo"),
            "mes": context.get("mes"),
            "dia": context.get("dia"),
        }

        # ========================================================
        # FOLLOW-UP
        # ========================================================

        if q.startswith("e "):

            if context.get("codigo_tipo"):
                filters["codigo_tipo"] = context["codigo_tipo"]

            if context.get("mes"):
                filters["mes"] = context["mes"]

            if context.get("dia"):
                filters["dia"] = context["dia"]

        # ========================================================
        # MES
        # ========================================================

        mes_match = re.search(
            r"m[eê]s\s+(\d{1,2})\s+de\s+(20\d{2})",
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
            r"dia\s+(\d{1,2})",
            q
        )

        if dia_match:
            filters["dia"] = dia_match.group(1).zfill(2)

        # ========================================================
        # ENTITY
        # ========================================================

        if (
            "change" in q
            or "changes" in q
            or "chg" in q
        ):
            filters["codigo_tipo"] = "CHG"

        if (
            "incidente" in q
            or "incidentes" in q
            or "inc" in q
        ):
            filters["codigo_tipo"] = "INC"

        # ========================================================
        # IC IMPACTADO
        # ========================================================

        ic_match = re.search(
            r"ic impactado\s+(.+)",
            q,
            re.IGNORECASE
        )

        ic_value = None

        if ic_match:

            ic_value = (
                ic_match.group(1)
                .replace("?", "")
                .strip()
            )

        # ========================================================
        # GRUPO
        # ========================================================

        grupo_match = re.search(
            r"grupo de atribui[cç][aã]o\s*=?\s*(.+)",
            q,
            re.IGNORECASE
        )

        grupo_value = None

        if grupo_match:

            grupo_value = (
                grupo_match.group(1)
                .replace("?", "")
                .strip()
            )

        # ========================================================
        # CHANGE REF
        # ========================================================

        change_ref_match = re.search(
            r"(CHG\d+)",
            q,
            re.IGNORECASE
        )

        change_ref = None

        if change_ref_match:
            change_ref = (
                change_ref_match.group(1)
                .upper()
            )

        # ========================================================
        # WHERE
        # ========================================================

        where = []

        if filters.get("codigo_tipo"):

            where.append(
                f"codigo_tipo = "
                f"'{filters['codigo_tipo']}'"
            )

        if filters.get("mes"):

            where.append(
                f"mes = '{filters['mes']}'"
            )

        if filters.get("dia"):

            where.append(
                f"dia = '{filters['dia']}'"
            )

        # ========================================================
        # CRÍTICO
        # ========================================================

        if filters.get("codigo_tipo") == "CHG":

            where.append(
                "numero LIKE 'CHG%'"
            )

        if filters.get("codigo_tipo") == "INC":

            where.append(
                "numero LIKE 'INC%'"
            )

        # ========================================================
        # IC
        # ========================================================

        if ic_value:

            where.append(
                f"""
                LOWER(ic_impactado)
                LIKE LOWER('%{ic_value}%')
                """
            )

        # ========================================================
        # GRUPO
        # ========================================================

        if grupo_value:

            where.append(
                f"""
                LOWER(grupo_de_atribuicao)
                LIKE LOWER('%{grupo_value}%')
                """
            )

        # ========================================================
        # APP
        # ========================================================

        if " app" in f" {q} ":

            where.append(
                "is_app = true"
            )

        # ========================================================
        # ECOMM
        # ========================================================

        if (
            "ecomm" in q
            or "e-commerce" in q
        ):

            where.append(
                "is_ecomm = true"
            )

        # ========================================================
        # CHANGE REF
        # ========================================================

        if change_ref:

            where.append(
                f"""
                (
                    LOWER(causado_pela_mudanca)
                    LIKE LOWER('%{change_ref}%')

                    OR LOWER(change_relacionada)
                    LIKE LOWER('%{change_ref}%')

                    OR LOWER(descricao)
                    LIKE LOWER('%{change_ref}%')
                )
                """
            )

        # ========================================================
        # SQL
        # ========================================================

        where_sql = " AND ".join(where)

        if where_sql:
            where_sql = "WHERE " + where_sql

        # ========================================================
        # LIST
        # ========================================================

        is_list = any(
            x in q
            for x in [
                "quais",
                "liste",
                "listar",
                "mostre",
                "códigos",
                "codigos",
            ]
        )

        # ========================================================
        # COUNT
        # ========================================================

        is_count = any(
            x in q
            for x in [
                "quantos",
                "quantas",
                "quantidade",
            ]
        )

        # ========================================================
        # LAST CODES
        # ========================================================

        if (
            "quais os códigos" in q
            or "quais os codigos" in q
        ):

            codes = context.get(
                "last_codes",
                []
            )

            return {
                "fallback_to_rag": False,
                "answer_text": "\n".join(codes),
                "route": "structured_memory"
            }

        # ========================================================
        # AURA WHATSAPP
        # ========================================================

        if (
            "aura whatsapp" in q
            and context.get("last_codes")
        ):

            code_list = context["last_codes"]

            in_clause = ",".join(
                f"'{x}'"
                for x in code_list
            )

            sql = f"""
                SELECT DISTINCT numero
                FROM knowledge
                WHERE numero IN ({in_clause})
                AND LOWER(ic_impactado)
                LIKE LOWER('%aura whatsapp%')
            """

            rows = conn.execute(sql).fetchall()

            found = [r[0] for r in rows]

            if found:

                return {
                    "fallback_to_rag": False,
                    "answer_text":
                        "Sim: " + ", ".join(found),
                    "route": "structured"
                }

            return {
                "fallback_to_rag": False,
                "answer_text": "Não",
                "route": "structured"
            }

        # ========================================================
        # COUNT
        # ========================================================

        if is_count and not is_list:

            sql = f"""
                SELECT COUNT(
                    DISTINCT numero
                ) AS total
                FROM knowledge
                {where_sql}
            """

            result = conn.execute(sql).fetchone()

            total = int(result[0] or 0)

            self.memory[case_id] = filters

            return {
                "fallback_to_rag": False,
                "route": "structured_count",
                "answer_text": str(total),
                "technical": {
                    "sql": sql,
                    "filters": filters,
                }
            }

        # ========================================================
        # LIST
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
            self.memory["case_id"]["last_codes"] = values

            return {
                "fallback_to_rag": False,
                "route": "structured_list",
                "answer_text": "\n".join(values),
                "technical": {
                    "sql": sql,
                    "count": len(values),
                }
            }

        # ========================================================
        # FALLBACK
        # ========================================================

        return {
            "fallback_to_rag": True
        }