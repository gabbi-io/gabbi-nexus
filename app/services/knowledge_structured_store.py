from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    import duckdb
    import pandas as pd

    HAS_DUCKDB = True
except Exception:
    duckdb = None
    pd = None
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


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = _norm(value)
    if text in {"true", "1", "sim", "yes", "y"}:
        return True
    if text in {"false", "0", "nao", "não", "no", "n", ""}:
        return False
    return None


def _extract_field(text: str, label: str) -> str:
    """
    Extrai campo no formato:
    Label: valor
    até a próxima linha que pareça outro campo.
    """
    if not text:
        return ""

    pattern = rf"(?im)^\s*{re.escape(label)}\s*:\s*(.+?)(?=\n[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-Za-zÀ-ÿ0-9 _/\-()]+:\s|\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return ""

    value = match.group(1).strip()
    value = re.sub(r"\n\s+", "\n", value)
    return value.strip()


def _extract_numero(text: str) -> str:
    direct = _extract_field(text, "Número")
    if direct:
        return _upper(direct)

    match = re.search(r"\b(CHG\d{5,}|INC\d{5,})\b", text or "", re.IGNORECASE)
    if match:
        return _upper(match.group(1))

    return ""


def _extract_mes(text: str) -> str:
    direct = _extract_field(text, "Mês")
    if direct:
        m = re.search(r"(20\d{2})[-/](\d{1,2})", direct)
        if m:
            return f"{m.group(1)}-{m.group(2).zfill(2)}"

    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/]\d{1,2}", text or "")
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}"

    return ""


def _extract_day(value: str) -> str:
    m = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", value or "")
    if not m:
        return ""
    return m.group(3).zfill(2)


def _sql_escape(value: Any) -> str:
    return _safe(value).replace("'", "''")


class KnowledgeStructuredStore:
    """
    Motor estruturado determinístico para CHG/INC.

    Regras importantes:
    - CHG só é CHG se Número começar com CHG.
    - INC só é INC se Número começar com INC.
    - Menção textual a CHG dentro de incidente não vira change.
    - Para contagem/lista usa DISTINCT numero.
    - Perguntas analíticas não devem cair em RAG.
    """

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
        "incidente",
        "incidentes",
        "app",
        "ecomm",
        "aura",
        "whatsapp",
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

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "reason": "duckdb_not_installed",
                "install": "pip install duckdb pandas",
            }

        try:
            with self._connect() as con:
                total = con.execute("SELECT COUNT(*) FROM knowledge_articles_structured").fetchone()[0]
                cases = con.execute("SELECT COUNT(DISTINCT case_id) FROM knowledge_articles_structured").fetchone()[0]
            return {
                "enabled": True,
                "db_path": str(self.db_path),
                "rows": int(total),
                "cases": int(cases),
            }
        except Exception as exc:
            return {
                "enabled": False,
                "db_path": str(self.db_path),
                "error": str(exc),
            }

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("DuckDB não está instalado.")
        return duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_articles_structured (
                    case_id TEXT,
                    knowledge_version TEXT,
                    article_id TEXT,
                    ref_id TEXT,
                    topic_id TEXT,
                    topic_name TEXT,
                    categoria TEXT,
                    numero TEXT,
                    codigo_tipo TEXT,
                    mes TEXT,
                    dia TEXT,
                    estado TEXT,
                    tipo TEXT,
                    prioridade TEXT,
                    ic_impactado TEXT,
                    grupo_atribuicao TEXT,
                    canal_impactado TEXT,
                    descricao_resumida TEXT,
                    descricao TEXT,
                    causado_pela_mudanca TEXT,
                    data_inicio_planejada TEXT,
                    data_termino_planejada TEXT,
                    aberto TEXT,
                    resolvido TEXT,
                    encerrado TEXT,
                    is_app BOOLEAN,
                    is_ecomm BOOLEAN,
                    raw_text TEXT,
                    raw_json TEXT
                )
                """
            )

    def replace_case_rows(
        self,
        case_id: str,
        rows: list[dict],
        knowledge_version: str | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"success": False, "reason": "duckdb_not_installed"}

        normalized: list[dict[str, Any]] = []

        for row in rows or []:
            record = self._row_to_record(case_id, row, knowledge_version)
            if record.get("numero"):
                normalized.append(record)

        with self._connect() as con:
            con.execute(
            "DELETE FROM knowledge_articles_structured WHERE case_id = ?",
            [case_id],
            )

            if normalized:
                df = pd.DataFrame(normalized)
                con.register("rows_df", df)

                con.execute(
                    """
                    INSERT INTO knowledge_articles_structured (
                        case_id,
                        knowledge_version,
                        article_id,
                        ref_id,
                        topic_id,
                        topic_name,
                        categoria,
                        numero,
                        codigo_tipo,
                        mes,
                        dia,
                        estado,
                        tipo,
                        prioridade,
                        ic_impactado,
                        grupo_atribuicao,
                        canal_impactado,
                        descricao_resumida,
                        descricao,
                        causado_pela_mudanca,
                        data_inicio_planejada,
                        data_termino_planejada,
                        aberto,
                        resolvido,
                        encerrado,
                        is_app,
                        is_ecomm,
                        raw_text,
                        raw_json
                    )
                    SELECT
                        case_id,
                        knowledge_version,
                        article_id,
                        ref_id,
                        topic_id,
                        topic_name,
                        categoria,
                        numero,
                        codigo_tipo,
                        mes,
                        dia,
                        estado,
                        tipo,
                        prioridade,
                        ic_impactado,
                        grupo_atribuicao,
                        canal_impactado,
                        descricao_resumida,
                        descricao,
                        causado_pela_mudanca,
                        data_inicio_planejada,
                        data_termino_planejada,
                        aberto,
                        resolvido,
                        encerrado,
                        is_app,
                        is_ecomm,
                        raw_text,
                        raw_json
                    FROM rows_df
                    """
                )
                df = pd.DataFrame(normalized)
                con.register("rows_df", df)
                con.execute(
                    """
                    INSERT INTO knowledge_articles_structured
                    SELECT
                        case_id,
                        knowledge_version,
                        article_id,
                        ref_id,
                        topic_id,
                        topic_name,
                        categoria,
                        numero,
                        codigo_tipo,
                        mes,
                        dia,
                        estado,
                        tipo,
                        prioridade,
                        ic_impactado,
                        grupo_atribuicao,
                        canal_impactado,
                        descricao_resumida,
                        descricao,
                        causado_pela_mudanca,
                        data_inicio_planejada,
                        data_termino_planejada,
                        aberto,
                        resolvido,
                        encerrado,
                        is_app,
                        is_ecomm,
                        raw_text,
                        raw_json
                    FROM rows_df
                    """
                )

        self.memory.pop(case_id, None)

        return {
            "success": True,
            "case_id": case_id,
            "rows_received": len(rows or []),
            "rows_saved": len(normalized),
            "knowledge_version": knowledge_version,
        }

    def _row_to_record(
        self,
        case_id: str,
        row: dict[str, Any],
        knowledge_version: str | None,
    ) -> dict[str, Any]:
        raw_json = json.dumps(row, ensure_ascii=False, default=str)

        document = row.get("document")
        if isinstance(document, str):
            try:
                document_obj = json.loads(document)
            except Exception:
                document_obj = {}
        elif isinstance(document, dict):
            document_obj = document
        else:
            document_obj = {}

        article_text = _safe(row.get("article") or row.get("text") or "")
        raw_text = article_text or raw_json

        numero = (
            _upper(document_obj.get("Número"))
            or _upper(document_obj.get("numero"))
            or _upper(row.get("numero"))
            or _upper(row.get("Número"))
            or _extract_numero(article_text)
        )

        categoria = (
            _upper(document_obj.get("categoria"))
            or _upper(document_obj.get("Categoria"))
            or _upper(row.get("categoria"))
            or _upper(row.get("Categoria"))
        )

        if numero.startswith("CHG"):
            codigo_tipo = "CHG"
        elif numero.startswith("INC"):
            codigo_tipo = "INC"
        elif categoria in {"CHG", "INC"}:
            codigo_tipo = categoria
        else:
            codigo_tipo = "OTHER"

        mes = (
            _safe(document_obj.get("mes"))
            or _safe(document_obj.get("Mês"))
            or _safe(row.get("mes"))
            or _safe(row.get("Mês"))
            or _extract_mes(article_text)
        )
        m = re.search(r"(20\d{2})[-/](\d{1,2})", mes)
        mes = f"{m.group(1)}-{m.group(2).zfill(2)}" if m else ""

        data_inicio = (
            _safe(document_obj.get("Data de início planejada"))
            or _safe(document_obj.get("data_inicio_planejada"))
            or _safe(row.get("data_inicio_planejada"))
            or _extract_field(article_text, "Data de início planejada")
        )

        aberto = (
            _safe(document_obj.get("Aberto"))
            or _safe(document_obj.get("aberto"))
            or _safe(row.get("Aberto"))
            or _safe(row.get("aberto"))
            or _extract_field(article_text, "Aberto")
        )

        if codigo_tipo == "CHG":
            dia = _extract_day(data_inicio)
        elif codigo_tipo == "INC":
            dia = _extract_day(aberto)
        else:
            dia = _extract_day(data_inicio or aberto)

        if not mes:
            base_date = data_inicio or aberto
            mm = re.search(r"(20\d{2})[-/](\d{1,2})", base_date)
            if mm:
                mes = f"{mm.group(1)}-{mm.group(2).zfill(2)}"

        ic = (
            _safe(document_obj.get("IC Impactado"))
            or _safe(document_obj.get("ic_impactado"))
            or _safe(row.get("ic_impactado"))
            or _extract_field(article_text, "IC Impactado")
        )

        grupo = (
            _safe(document_obj.get("Grupo de atribuição"))
            or _safe(document_obj.get("grupo_atribuicao"))
            or _safe(document_obj.get("grupo_de_atribuicao"))
            or _safe(row.get("grupo_atribuicao"))
            or _safe(row.get("grupo_de_atribuicao"))
            or _extract_field(article_text, "Grupo de atribuição")
        )

        canal = (
            _safe(document_obj.get("Canal impactado"))
            or _safe(document_obj.get("Canal Impactado"))
            or _safe(document_obj.get("canal_impactado"))
            or _safe(row.get("canal_impactado"))
            or _extract_field(article_text, "Canal impactado")
            or _extract_field(article_text, "Canal Impactado")
        )

        descricao_resumida = (
            _safe(document_obj.get("Descrição resumida"))
            or _safe(document_obj.get("descricao_resumida"))
            or _safe(row.get("descricao_resumida"))
            or _extract_field(article_text, "Descrição resumida")
        )

        descricao = (
            _safe(document_obj.get("Descrição"))
            or _safe(document_obj.get("descricao"))
            or _safe(row.get("descricao"))
            or _extract_field(article_text, "Descrição")
        )

        causado = (
            _safe(document_obj.get("Causado pela mudança"))
            or _safe(document_obj.get("causado_pela_mudanca"))
            or _safe(row.get("causado_pela_mudanca"))
            or _extract_field(article_text, "Causado pela mudança")
        )

        is_app_raw = (
            document_obj.get("is_app")
            if "is_app" in document_obj
            else row.get("is_app")
        )
        is_ecomm_raw = (
            document_obj.get("is_ecomm")
            if "is_ecomm" in document_obj
            else row.get("is_ecomm")
        )

        is_app = _bool_value(is_app_raw)
        is_ecomm = _bool_value(is_ecomm_raw)

        search_blob = _norm(" ".join([canal, descricao_resumida, descricao, raw_text]))

        if is_app is None:
            is_app = bool(
                "app vivo" in search_blob
                or "aplicativo vivo" in search_blob
                or re.search(r"\bapp\b", search_blob)
            )

        if is_ecomm is None:
            is_ecomm = bool(
                "ecomm" in search_blob
                or "e commerce" in search_blob
                or "ecommerce" in search_blob
                or "loja online" in search_blob
            )

        return {
            "case_id": case_id,
            "knowledge_version": knowledge_version,
            "article_id": _safe(row.get("article_id") or row.get("id")),
            "ref_id": _safe(row.get("ref_id") or row.get("refId")),
            "topic_id": _safe(row.get("topic_id") or row.get("topicId")),
            "topic_name": _safe(row.get("topic_name") or row.get("topicName")),
            "categoria": categoria or codigo_tipo,
            "numero": numero,
            "codigo_tipo": codigo_tipo,
            "mes": mes,
            "dia": dia,
            "estado": _safe(document_obj.get("Estado") or document_obj.get("estado") or _extract_field(article_text, "Estado")),
            "tipo": _safe(document_obj.get("Tipo") or document_obj.get("tipo") or _extract_field(article_text, "Tipo")),
            "prioridade": _safe(document_obj.get("Prioridade") or document_obj.get("prioridade") or _extract_field(article_text, "Prioridade")),
            "ic_impactado": ic,
            "grupo_atribuicao": grupo,
            "canal_impactado": canal,
            "descricao_resumida": descricao_resumida,
            "descricao": descricao,
            "causado_pela_mudanca": causado,
            "data_inicio_planejada": data_inicio,
            "data_termino_planejada": _safe(document_obj.get("Data de término planejada") or _extract_field(article_text, "Data de término planejada")),
            "aberto": aberto,
            "resolvido": _safe(document_obj.get("Resolvido") or _extract_field(article_text, "Resolvido")),
            "encerrado": _safe(document_obj.get("Encerrado") or _extract_field(article_text, "Encerrado")),
            "is_app": bool(is_app),
            "is_ecomm": bool(is_ecomm),
            "raw_text": raw_text,
            "raw_json": raw_json,
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

        if not self._is_analytic(q):
            return {"fallback_to_rag": True}

        try:
            result = self._answer_structured(case_id, question)
            return result
        except Exception as exc:
            return {
                "fallback_to_rag": False,
                "route": "structured_error",
                "query_type": "error",
                "answer_text": f"Erro ao consultar base estruturada: {exc}",
                "technical": {"error": str(exc)},
            }

    def _answer_structured(self, case_id: str, question: str) -> dict[str, Any]:
        q = _norm(question)
        context = dict(self.memory.get(case_id) or {})

        if "quais os codigos" in q or "quais os códigos" in question.lower():
            codes = context.get("last_codes") or []
            return {
                "fallback_to_rag": False,
                "route": "structured_memory",
                "query_type": "list",
                "answer_text": "\n".join(codes) if codes else "Não há códigos em memória para listar.",
                "technical": {"memory": context},
            }

        if "aura whatsapp" in q and context.get("last_codes"):
            return self._answer_aura_whatsapp(case_id, context)

        plan = self._build_plan(question, context)
        where_sql, params = self._build_where(plan)

        is_count = any(x in q for x in ["quantos", "quantas", "quantidade", "qtd", "qtde", "total"])
        is_list = any(x in q for x in ["quais", "qual", "liste", "listar", "lista", "mostre", "codigos", "códigos"])

        if plan.get("force_list"):
            is_list = True
            is_count = False

        with self._connect() as con:
            if is_count and not is_list:
                sql = f"""
                    SELECT COUNT(DISTINCT numero) AS total
                    FROM knowledge_articles_structured
                    {where_sql}
                """
                total = int(con.execute(sql, params).fetchone()[0] or 0)

                codes_sql = f"""
                    SELECT DISTINCT numero
                    FROM knowledge_articles_structured
                    {where_sql}
                    ORDER BY numero
                """
                codes = [r[0] for r in con.execute(codes_sql, params).fetchall()]

                self._save_memory(case_id, plan, codes)

                return {
                    "fallback_to_rag": False,
                    "route": "structured_count",
                    "query_type": "count",
                    "answer_text": str(total),
                    "technical": {"sql": sql, "params": params, "plan": plan, "count": total},
                }

            if is_list:
                sql = f"""
                    SELECT DISTINCT numero
                    FROM knowledge_articles_structured
                    {where_sql}
                    ORDER BY numero
                """
                codes = [r[0] for r in con.execute(sql, params).fetchall()]
                self._save_memory(case_id, plan, codes)

                return {
                    "fallback_to_rag": False,
                    "route": "structured_list",
                    "query_type": "list",
                    "answer_text": "\n".join(codes) if codes else "Nenhum registro encontrado.",
                    "technical": {"sql": sql, "params": params, "plan": plan, "count": len(codes)},
                }

        return {
            "fallback_to_rag": False,
            "route": "structured_no_answer",
            "query_type": "unknown",
            "answer_text": "Não consegui calcular com segurança pela base estruturada.",
            "technical": {"plan": plan},
        }

    def _build_plan(self, question: str, context: dict[str, Any]) -> dict[str, Any]:
        q = _norm(question)
        plan: dict[str, Any] = {}

        followup = q.startswith("e ") or any(x in q for x in ["destes", "destas", "desses", "dessas", "deles", "delas"])

        if followup:
            for key in ["codigo_tipo", "mes", "dia", "is_app", "is_ecomm"]:
                if context.get(key) is not None:
                    plan[key] = context.get(key)

        if any(x in q for x in ["change", "changes", "chg", "mudanca", "mudança"]):
            plan["codigo_tipo"] = "CHG"

        if any(x in q for x in ["incidente", "incidentes", "inc"]):
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
            if "referencia" in q or "referência" in q or "causado" in q:
                plan["codigo_tipo"] = plan.get("codigo_tipo") or "INC"

        if re.search(r"\bapp\b", q):
            plan["is_app"] = True

        if any(x in q for x in ["ecomm", "e commerce", "e-commerce", "ecommerce"]):
            plan["is_ecomm"] = True

        if "grupo_atribuicao" in plan and not any(x in q for x in ["quais", "liste", "listar", "codigos", "códigos"]):
            plan["force_count"] = True

        if plan.get("force_count"):
            plan.pop("force_list", None)

        return plan

    def _build_where(self, plan: dict[str, Any]) -> tuple[str, list[Any]]:
        where = ["case_id = ?"]
        params: list[Any] = []

        # case_id entra no answer
        # será substituído abaixo
        # técnica simples: primeiro param será preenchido pelo caller
        # mas aqui precisamos do case no caller; então ajustado no final via closure
        # para manter simples, será tratado fora? Não. Vamos inserir no começo depois.
        # Este método recebe plan sem case, então removemos case_id aqui.
        where = []
        params = []

        codigo_tipo = plan.get("codigo_tipo")
        if codigo_tipo:
            where.append("codigo_tipo = ?")
            params.append(codigo_tipo)

            if codigo_tipo == "CHG":
                where.append("numero LIKE 'CHG%'")
                where.append("categoria = 'CHG'")
            elif codigo_tipo == "INC":
                where.append("numero LIKE 'INC%'")
                where.append("categoria = 'INC'")

        if plan.get("mes"):
            where.append("mes = ?")
            params.append(plan["mes"])

        if plan.get("dia"):
            where.append("dia = ?")
            params.append(plan["dia"])

        if plan.get("ic_impactado"):
            where.append("LOWER(ic_impactado) LIKE LOWER(?)")
            params.append(f"%{plan['ic_impactado']}%")

        if plan.get("grupo_atribuicao"):
            where.append("LOWER(grupo_atribuicao) LIKE LOWER(?)")
            params.append(f"%{plan['grupo_atribuicao']}%")

        if plan.get("is_app") is True:
            where.append("is_app = TRUE")

        if plan.get("is_ecomm") is True:
            where.append("is_ecomm = TRUE")

        if plan.get("change_ref"):
            where.append(
                """
                (
                    LOWER(causado_pela_mudanca) LIKE LOWER(?)
                    OR LOWER(descricao) LIKE LOWER(?)
                    OR LOWER(descricao_resumida) LIKE LOWER(?)
                    OR LOWER(raw_text) LIKE LOWER(?)
                )
                """
            )
            ref = f"%{plan['change_ref']}%"
            params.extend([ref, ref, ref, ref])

        if not where:
            where_sql = ""
        else:
            where_sql = "WHERE " + " AND ".join(where)

        return where_sql, params

    def _save_memory(self, case_id: str, plan: dict[str, Any], codes: list[str]) -> None:
        memory = dict(self.memory.get(case_id) or {})
        for key in ["codigo_tipo", "mes", "dia", "is_app", "is_ecomm"]:
            if key in plan:
                memory[key] = plan[key]
        memory["last_codes"] = codes
        self.memory[case_id] = memory

    def _answer_aura_whatsapp(self, case_id: str, context: dict[str, Any]) -> dict[str, Any]:
        codes = context.get("last_codes") or []
        if not codes:
            return {
                "fallback_to_rag": False,
                "route": "structured_memory",
                "query_type": "boolean",
                "answer_text": "Não",
            }

        placeholders = ",".join(["?"] * len(codes))
        params = list(codes)

        sql = f"""
            SELECT DISTINCT numero
            FROM knowledge_articles_structured
            WHERE numero IN ({placeholders})
            AND (
                LOWER(ic_impactado) LIKE LOWER('%AURA WHATSAPP%')
                OR LOWER(descricao) LIKE LOWER('%AURA WHATSAPP%')
                OR LOWER(descricao_resumida) LIKE LOWER('%AURA WHATSAPP%')
                OR LOWER(raw_text) LIKE LOWER('%AURA WHATSAPP%')
            )
            ORDER BY numero
        """

        with self._connect() as con:
            found = [r[0] for r in con.execute(sql, params).fetchall()]

        if found:
            return {
                "fallback_to_rag": False,
                "route": "structured_boolean",
                "query_type": "boolean",
                "answer_text": "Sim: " + ", ".join(found),
                "technical": {"sql": sql, "codes": codes, "found": found},
            }

        return {
            "fallback_to_rag": False,
            "route": "structured_boolean",
            "query_type": "boolean",
            "answer_text": "Não",
            "technical": {"sql": sql, "codes": codes, "found": []},
        }

    def _extract_month_from_question(self, q: str) -> str:
        patterns = [
            r"m[eê]s\s+(\d{1,2})\s+de\s+(20\d{2})",
            r"\b(\d{1,2})\s+de\s+(20\d{2})\b",
            r"\b(20\d{2})[-/](\d{1,2})\b",
        ]

        for pattern in patterns:
            m = re.search(pattern, q)
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

    def _extract_day_from_question(self, q: str) -> str:
        m = re.search(r"\bdia\s+(\d{1,2})\b", q)
        if not m:
            return ""
        day = m.group(1).zfill(2)
        if 1 <= int(day) <= 31:
            return day
        return ""

    def _extract_after(self, question: str, pattern: str) -> str:
        m = re.search(pattern, question, re.IGNORECASE)
        if not m:
            return ""

        value = m.group(1).strip()
        value = value.replace("?", "").strip()

        stop_words = [
            "\n",
            " e grupo ",
            " e com ",
            " e que ",
        ]

        lower = value.lower()
        cut = len(value)
        for stop in stop_words:
            idx = lower.find(stop)
            if idx >= 0:
                cut = min(cut, idx)

        return value[:cut].strip()

    def _is_analytic(self, q: str) -> bool:
        return any(term in q for term in self.ANALYTIC_TERMS)