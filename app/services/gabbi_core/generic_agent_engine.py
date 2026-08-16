from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


def _safe(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _safe(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
    return " ".join(text.split())


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _sql_ident(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


@dataclass
class AgentSchemaField:
    name: str
    source: str = "column"  # column | raw_json | document_json | text
    type: str = "TEXT"
    aliases: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class GenericQueryPlan:
    route: str = "semantic"  # semantic | analytic | hybrid | legacy_vivo
    intent: str = "answer"   # count | list | detail | group | rank | summary | answer
    entity: str = ""
    metric: str = "count"
    filters: list[dict[str, Any]] = field(default_factory=list)
    group_by: str | None = None
    sort_by: str | None = None
    sort_direction: str = "desc"
    limit: int = 20
    confidence: float = 0.0
    reason: str = ""


class GenericAgentAnswerEngine:
    """Motor genérico para agentes de conhecimento.

    Este componente remove o acoplamento do núcleo com Vivo/ITSM.
    Ele usa o schema real do case/agente para decidir se a pergunta é:
      - semântica/documental: deixa o RAG responder;
      - analítica: executa SQL controlado sobre campos permitidos;
      - híbrida: executa dados e devolve material para narrativa/RAG.

    As regras específicas de Vivo ficam preservadas no KnowledgeStructuredStore legado.
    Quando a pergunta tem forte marcador ITSM/Vivo, este motor retorna legacy_vivo
    para deixar as otimizações antigas responderem.
    """

    GENERIC_ANALYTIC_MARKERS = {
        "quantos", "quantas", "quantidade", "total", "qtd", "qtde", "conte", "contar",
        "liste", "listar", "quais", "qual", "mostre", "traga", "exiba", "detalhe", "detalhar",
        "ranking", "top", "maior", "menor", "mais", "menos", "agrupe", "agrupar", "distribuicao",
        "distribuição", "por", "media", "média", "soma", "somatorio", "somatório", "percentual",
        "porcentagem", "comparar", "compare", "comparativo",
    }

    SEMANTIC_MARKERS = {
        "explique", "explica", "como funciona", "o que é", "o que e", "defina", "resuma", "resumo",
        "fale sobre", "sobre o que", "qual o teor", "descreva", "contexto", "orientação", "orientacao",
        "passo a passo", "procedimento", "manual", "regra", "política", "politica", "treinamento",
    }

    # Marcadores que indicam que as rotas antigas do agente Vivo são mais assertivas.
    VIVO_ITSM_MARKERS = {
        "incidente", "incidentes", "inc", "change", "changes", "chg", "mttr", "p1", "p2", "p3",
        "app vivo", "meu vivo", "esim", "recarga", "ic impactado", "grupo de atribuicao",
        "grupo de atribuição", "parada sistemica", "parada sistêmica", "tempo de impacto",
        "causado pela mudança", "causado pela mudanca",
    }

    STOPWORDS = {
        "o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "em", "no", "na", "nos", "nas",
        "por", "para", "pra", "com", "sem", "e", "ou", "que", "qual", "quais", "quantos", "quantas",
        "total", "quantidade", "me", "traga", "mostre", "liste", "detalhe", "sobre", "esse", "essa", "esses",
        "essas", "ele", "ela", "eles", "elas", "teve", "tem", "tivemos", "existe", "existem",
    }

    BASE_ALIASES = {
        "id": ["id", "codigo", "código", "numero", "número", "identificador"],
        "name": ["nome", "name", "titulo", "título"],
        "title": ["titulo", "título", "assunto", "tema"],
        "description": ["descricao", "descrição", "detalhe", "conteudo", "conteúdo", "texto"],
        "date": ["data", "dia", "mes", "mês", "competencia", "competência", "periodo", "período"],
        "status": ["status", "estado", "situacao", "situação"],
        "category": ["categoria", "tipo", "classe", "grupo"],
        "priority": ["prioridade", "severidade", "criticidade"],
        "owner": ["responsavel", "responsável", "dono", "owner", "atribuido", "atribuído"],
        "value": ["valor", "preco", "preço", "custo", "montante", "total"],
    }

    MONTHS = {
        "janeiro": "01", "jan": "01", "fevereiro": "02", "fev": "02", "marco": "03", "março": "03", "mar": "03",
        "abril": "04", "abr": "04", "maio": "05", "junho": "06", "jun": "06", "julho": "07", "jul": "07",
        "agosto": "08", "ago": "08", "setembro": "09", "set": "09", "outubro": "10", "out": "10",
        "novembro": "11", "nov": "11", "dezembro": "12", "dez": "12",
    }

    def __init__(self, store: Any):
        self.store = store
        self.enabled = os.getenv("GABBI_GENERIC_AGENT_ENGINE", "true").strip().lower() in {"1", "true", "yes", "sim", "on"}
        self.use_vivo_legacy = os.getenv("GABBI_ENABLE_VIVO_SPECIALIZATION", "true").strip().lower() in {"1", "true", "yes", "sim", "on"}
        self.max_sample_rows = int(os.getenv("GABBI_GENERIC_SCHEMA_SAMPLE", "300"))

    def answer_question(self, case_id: str, question: str, chat_history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        q = _norm(question)
        if not q:
            return {"fallback_to_rag": True, "route": "generic_empty_question"}

        schema = self.discover_schema(case_id)
        plan = self.plan(case_id, question, schema, chat_history)

        # Mantém o que já funciona para Vivo/ITSM sem contaminar outros agentes.
        if plan.route == "legacy_vivo":
            return None

        if plan.route == "semantic":
            self._save_context(case_id, plan, None)
            return {
                "fallback_to_rag": True,
                "route": "generic_rag_semantic",
                "query_type": "semantic",
                "generic_engine": True,
                "agent_schema": self._schema_public(schema),
                "reason": plan.reason,
            }

        if plan.route in {"analytic", "hybrid"}:
            result = self.execute_plan(case_id, plan, schema)
            self._save_context(case_id, plan, result)
            if plan.route == "hybrid":
                result["fallback_to_rag"] = True
                result["route"] = "generic_hybrid_sql_plus_rag"
                result["generic_engine"] = True
                result["structured_context"] = result.get("data")
                return result
            return result

        return {"fallback_to_rag": True, "route": "generic_unknown_route", "generic_engine": True}

    # ------------------------------------------------------------------
    # Schema discovery
    # ------------------------------------------------------------------
    def discover_schema(self, case_id: str) -> dict[str, AgentSchemaField]:
        fields: dict[str, AgentSchemaField] = {}
        if not getattr(self.store, "enabled", False):
            return fields
        try:
            with self.store._connect() as con:
                cols = self.store._table_cols(con)
                for col in cols:
                    if col in {"raw_json", "article_text", "updated_at"}:
                        continue
                    fields[col] = AgentSchemaField(name=col, source="column", aliases=self._aliases_for(col))

                rows = con.execute(
                    f"SELECT raw_json, article_text FROM {self.store.TABLE} WHERE case_id = ? LIMIT ?",
                    [case_id, self.max_sample_rows],
                ).fetchall()
        except Exception:
            return fields

        for raw_json, article_text in rows:
            for key, value in self._flatten_json_keys(raw_json).items():
                nkey = self._safe_field_name(key)
                if not nkey or nkey in fields:
                    continue
                fields[nkey] = AgentSchemaField(name=nkey, source="raw_json", type=self._infer_type(value), aliases=self._aliases_for(nkey), confidence=0.85)
            # campos textuais no formato "Label: valor"
            for label in self._labels_from_text(article_text):
                nkey = self._safe_field_name(label)
                if nkey and nkey not in fields:
                    fields[nkey] = AgentSchemaField(name=nkey, source="text", type="TEXT", aliases=self._aliases_for(nkey), confidence=0.65)
        return fields

    def _flatten_json_keys(self, raw_json: Any) -> dict[str, Any]:
        try:
            obj = json.loads(raw_json) if isinstance(raw_json, str) else raw_json
        except Exception:
            return {}
        if not isinstance(obj, dict):
            return {}
        out: dict[str, Any] = {}

        def walk(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for k, v in value.items():
                    key = f"{prefix}.{k}" if prefix else str(k)
                    walk(key, v)
            else:
                out[prefix] = value

        walk("", obj)
        # Se document vier como JSON serializado, indexa também.
        doc = obj.get("document")
        if isinstance(doc, str):
            try:
                doc_obj = json.loads(doc)
                if isinstance(doc_obj, dict):
                    for k, v in doc_obj.items():
                        out[f"document.{k}"] = v
            except Exception:
                pass
        return out

    def _labels_from_text(self, text: Any) -> list[str]:
        labels: list[str] = []
        for line in _safe(text).splitlines():
            m = re.match(r"^\s*([^:]{2,80})\s*:\s*.+$", line)
            if m:
                labels.append(m.group(1).strip())
        return labels[:80]

    def _safe_field_name(self, value: str) -> str:
        text = _norm(value).replace(".", "_").replace("-", "_").replace("/", "_")
        text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
        return text[:80]

    def _infer_type(self, value: Any) -> str:
        if isinstance(value, bool):
            return "BOOLEAN"
        if isinstance(value, int):
            return "BIGINT"
        if isinstance(value, float):
            return "DOUBLE"
        return "TEXT"

    def _aliases_for(self, field_name: str) -> list[str]:
        aliases = {field_name, field_name.replace("_", " ")}
        compact = _compact(field_name)
        for canonical, vals in self.BASE_ALIASES.items():
            if compact == _compact(canonical) or any(_compact(v) in compact or compact in _compact(v) for v in vals):
                aliases.update(vals)
        return sorted(aliases)

    # ------------------------------------------------------------------
    # Planner
    # ------------------------------------------------------------------
    def plan(self, case_id: str, question: str, schema: dict[str, AgentSchemaField], chat_history: list[dict[str, Any]] | None = None) -> GenericQueryPlan:
        q = _norm(question)
        tokens = set(q.split())

        if self.use_vivo_legacy and self._looks_like_vivo_itsm(q, schema):
            return GenericQueryPlan(route="legacy_vivo", reason="vivo_itsm_specialization_preserved", confidence=0.95)

        semantic = any(m in q for m in self.SEMANTIC_MARKERS)
        analytic = bool(tokens & self.GENERIC_ANALYTIC_MARKERS) or self._mentions_schema_field(q, schema) or bool(self._extract_month(q))
        hybrid = semantic and analytic

        intent = "answer"
        if any(x in q for x in ["quantos", "quantas", "quantidade", "total", "qtd", "qtde", "conte", "contar"]):
            intent = "count"
        elif any(x in q for x in ["ranking", "top", "maior", "menor", "mais", "menos"]):
            intent = "rank"
        elif any(x in q for x in ["agrupe", "agrupar", "distribuicao", "distribuição", "por status", "por tipo", "por categoria"]):
            intent = "group"
        elif any(x in q for x in ["liste", "listar", "quais", "mostre", "traga", "exiba"]):
            intent = "list"
        elif any(x in q for x in ["detalhe", "detalhar", "descreva"]):
            intent = "detail"
        elif semantic:
            intent = "summary"

        if not analytic and semantic:
            return GenericQueryPlan(route="semantic", intent=intent, confidence=0.9, reason="semantic_question")
        if not analytic:
            return GenericQueryPlan(route="semantic", intent="answer", confidence=0.65, reason="no_analytic_marker")

        filters = self._extract_filters(q, schema)
        month = self._extract_month(q)
        if month and "mes" in schema and not any(f.get("field") == "mes" for f in filters):
            filters.append({"field": "mes", "operator": "eq", "value": month})

        group_by = self._extract_group_by(q, schema) if intent == "group" else None
        sort_by = self._extract_sort_by(q, schema) if intent == "rank" else None
        entity = self._extract_entity(q, schema)
        limit = 10 if intent == "rank" else 150 if intent in {"list", "detail"} else 20
        return GenericQueryPlan(
            route="hybrid" if hybrid else "analytic",
            intent=intent,
            entity=entity,
            metric="count" if intent in {"count", "group"} else "unknown",
            filters=filters,
            group_by=group_by,
            sort_by=sort_by,
            limit=limit,
            confidence=0.82,
            reason="generic_dynamic_schema_planner",
        )

    def _looks_like_vivo_itsm(self, q: str, schema: dict[str, AgentSchemaField]) -> bool:
        if any(m in q for m in self.VIVO_ITSM_MARKERS):
            return True
        # Se a pergunta usa campos muito específicos do modelo antigo, preserva legado.
        legacy_fields = {"numero", "codigo_tipo", "prioridade", "tempo_impacto", "tempo_impacto_segundos", "ic_impactado", "grupo_atribuicao"}
        return bool(legacy_fields.intersection(schema.keys()) and any(x in q for x in ["p1", "p2", "p3", "mttr", "impacto", "change", "incidente"]))

    def _mentions_schema_field(self, q: str, schema: dict[str, AgentSchemaField]) -> bool:
        return self._resolve_field(q, schema) is not None

    def _extract_filters(self, q: str, schema: dict[str, AgentSchemaField]) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        # Padrões: campo = valor, campo: valor, campo com valor, status pago
        for field_name, field in schema.items():
            aliases = sorted(field.aliases + [field_name, field_name.replace("_", " ")], key=len, reverse=True)
            for alias in aliases:
                a = re.escape(_norm(alias))
                patterns = [
                    rf"\b{a}\s*(?:=|:|igual a|como|com)\s*([a-z0-9_:/\-\. ]{{2,80}})",
                    rf"\b{a}\s+([a-z0-9_:/\-\.]{{2,80}})\b",
                ]
                for pat in patterns:
                    m = re.search(pat, q)
                    if not m:
                        continue
                    value = self._clean_value(m.group(1))
                    if value and value not in self.STOPWORDS:
                        filters.append({"field": field_name, "operator": "contains", "value": value})
                        break
                if any(f.get("field") == field_name for f in filters):
                    break
        # Códigos/IDs explícitos genéricos.
        for code in re.findall(r"\b[A-Z]{2,10}\d{3,}\b", question := q.upper()):
            id_field = self._best_id_field(schema)
            if id_field:
                filters.append({"field": id_field, "operator": "contains", "value": code})
        return self._dedupe_filters(filters)

    def _clean_value(self, value: str) -> str:
        text = _norm(value)
        stop_markers = [" e ", " ou ", " com ", " por ", " no ", " na ", " em ", " para "]
        for marker in stop_markers:
            if marker in text and len(text.split()) > 4:
                text = text.split(marker, 1)[0]
        return text.strip(" .,-")[:120]

    def _dedupe_filters(self, filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        seen = set()
        for f in filters:
            key = (f.get("field"), f.get("operator"), str(f.get("value")).lower())
            if key not in seen:
                seen.add(key)
                out.append(f)
        return out

    def _extract_month(self, q: str) -> str:
        m = re.search(r"(20\d{2})[-/](\d{1,2})", q)
        if m:
            return f"{m.group(1)}-{m.group(2).zfill(2)}"
        year = re.search(r"\b(20\d{2})\b", q)
        for name, num in self.MONTHS.items():
            if re.search(rf"\b{re.escape(name)}\b", q):
                return f"{year.group(1) if year else '2025'}-{num}"
        return ""

    def _resolve_field(self, text: str, schema: dict[str, AgentSchemaField]) -> str | None:
        text_norm = _norm(text)
        best: tuple[str, float] | None = None
        for name, field in schema.items():
            candidates = [name, name.replace("_", " ")] + field.aliases
            for cand in candidates:
                cand_norm = _norm(cand)
                if not cand_norm:
                    continue
                score = 0.0
                if re.search(rf"\b{re.escape(cand_norm)}\b", text_norm):
                    score = 1.0
                else:
                    score = SequenceMatcher(None, text_norm, cand_norm).ratio() - 0.35
                if score > 0.55 and (best is None or score > best[1]):
                    best = (name, score)
        return best[0] if best else None

    def _extract_group_by(self, q: str, schema: dict[str, AgentSchemaField]) -> str | None:
        m = re.search(r"\bpor\s+([a-z0-9_ ]{2,40})", q)
        if m:
            return self._resolve_field(m.group(1), schema)
        for preferred in ["status", "estado", "categoria", "tipo", "prioridade", "mes", "data"]:
            f = self._resolve_field(preferred, schema)
            if f and preferred in q:
                return f
        return None

    def _extract_sort_by(self, q: str, schema: dict[str, AgentSchemaField]) -> str | None:
        for preferred in ["tempo_impacto_segundos", "tempo", "impacto", "valor", "total", "quantidade", "data", "prioridade"]:
            f = self._resolve_field(preferred, schema)
            if f and preferred in q:
                return f
        # fallback para primeiro campo numérico
        for name, field in schema.items():
            if field.type in {"BIGINT", "DOUBLE", "INTEGER", "FLOAT"}:
                return name
        return None

    def _extract_entity(self, q: str, schema: dict[str, AgentSchemaField]) -> str:
        words = [w for w in q.split() if len(w) > 2 and w not in self.STOPWORDS]
        return " ".join(words[:5])

    def _best_id_field(self, schema: dict[str, AgentSchemaField]) -> str | None:
        for candidate in ["numero", "codigo", "id", "article_id", "codigo_principal", "document_numero"]:
            f = self._resolve_field(candidate, schema)
            if f:
                return f
        return next(iter(schema.keys()), None) if schema else None

    # ------------------------------------------------------------------
    # SQL executor controlado
    # ------------------------------------------------------------------
    def execute_plan(self, case_id: str, plan: GenericQueryPlan, schema: dict[str, AgentSchemaField]) -> dict[str, Any]:
        if not getattr(self.store, "enabled", False):
            return self._error("DuckDB/base estruturada indisponível.", plan)
        try:
            where_sql, params = self._build_where(case_id, plan, schema)
            table = self.store.TABLE
            if plan.intent == "count":
                sql = f"SELECT COUNT(*) AS total FROM {table} WHERE {where_sql}"
                with self.store._connect() as con:
                    total = con.execute(sql, params).fetchone()[0]
                return self._ok(plan, f"Total: {int(total)}", {"total": int(total)}, sql, params)

            if plan.intent == "group" and plan.group_by:
                expr = self._field_expr(plan.group_by, schema)
                if not expr:
                    return self._error(f"Campo de agrupamento não permitido: {plan.group_by}", plan)
                sql = f"SELECT {expr} AS grupo, COUNT(*) AS total FROM {table} WHERE {where_sql} GROUP BY 1 ORDER BY total DESC LIMIT ?"
                with self.store._connect() as con:
                    rows = con.execute(sql, params + [plan.limit]).fetchall()
                lines = ["Distribuição:"] + [f"- {r[0] or '(vazio)'}: {int(r[1])}" for r in rows]
                return self._ok(plan, "\n".join(lines), {"rows": [{"grupo": r[0], "total": int(r[1])} for r in rows]}, sql, params)

            # list/detail/rank
            select_fields = self._default_select_fields(schema)
            order_sql = ""
            if plan.intent == "rank" and plan.sort_by:
                expr = self._field_expr(plan.sort_by, schema)
                if expr:
                    order_sql = f" ORDER BY {expr} {plan.sort_direction.upper()} NULLS LAST"
            if not order_sql:
                order_sql = " ORDER BY updated_at DESC NULLS LAST" if "updated_at" in getattr(self.store, "BASE_COLUMNS", {}) else ""
            select_sql = ", ".join(f"{self._field_expr(f, schema)} AS {_sql_ident(f)}" for f in select_fields if self._field_expr(f, schema))
            if not select_sql:
                select_sql = "article_text"
            sql = f"SELECT {select_sql} FROM {table} WHERE {where_sql}{order_sql} LIMIT ?"
            with self.store._connect() as con:
                rows = con.execute(sql, params + [plan.limit]).fetchall()
                cols = [d[0] for d in con.description]
            data = [dict(zip(cols, row)) for row in rows]
            answer = self._format_rows(data, plan)
            return self._ok(plan, answer, {"rows": data, "count": len(data)}, sql, params)
        except Exception as exc:
            return self._error(f"Erro no executor genérico: {type(exc).__name__}: {exc}", plan)

    def _build_where(self, case_id: str, plan: GenericQueryPlan, schema: dict[str, AgentSchemaField]) -> tuple[str, list[Any]]:
        clauses = ["case_id = ?"]
        params: list[Any] = [case_id]
        for f in plan.filters:
            field = f.get("field")
            op = f.get("operator") or "contains"
            value = f.get("value")
            expr = self._field_expr(field, schema) if field else None
            if not expr:
                continue
            if op == "eq":
                clauses.append(f"LOWER(COALESCE(CAST({expr} AS VARCHAR), '')) = LOWER(?)")
                params.append(str(value))
            elif op == "in" and isinstance(value, list):
                placeholders = ",".join(["?"] * len(value))
                clauses.append(f"LOWER(COALESCE(CAST({expr} AS VARCHAR), '')) IN ({placeholders})")
                params.extend([str(v).lower() for v in value])
            elif op in {"is_true", "is_false"}:
                clauses.append(f"CAST({expr} AS BOOLEAN) IS {'TRUE' if op == 'is_true' else 'FALSE'}")
            else:
                clauses.append(f"LOWER(COALESCE(CAST({expr} AS VARCHAR), '')) LIKE LOWER(?)")
                params.append(f"%{value}%")
        return " AND ".join(clauses), params

    def _field_expr(self, field_name: str | None, schema: dict[str, AgentSchemaField]) -> str | None:
        if not field_name or field_name not in schema:
            return None
        field = schema[field_name]
        if field.source == "column":
            return _sql_ident(field.name)
        if field.source == "raw_json":
            # DuckDB JSON path. Se não existir, retorna NULL sem quebrar.
            raw_path = field.name.replace("_", ".")
            return f"json_extract_string(raw_json, '$.{raw_path}')"
        if field.source == "text":
            # Campos extraídos por label são consultáveis por texto bruto.
            return "article_text"
        return _sql_ident(field.name) if field.name else None

    def _default_select_fields(self, schema: dict[str, AgentSchemaField]) -> list[str]:
        preferred = [
            "numero", "codigo", "id", "article_id", "titulo", "title", "name", "nome", "mes", "data", "status",
            "estado", "categoria", "tipo", "prioridade", "descricao_resumida", "descricao", "article_text",
        ]
        out: list[str] = []
        for p in preferred:
            f = self._resolve_field(p, schema)
            if f and f not in out:
                out.append(f)
        for name in schema.keys():
            if name not in out and len(out) < 8:
                out.append(name)
        return out[:8]

    def _format_rows(self, rows: list[dict[str, Any]], plan: GenericQueryPlan) -> str:
        if not rows:
            return "Não encontrei registros compatíveis com os filtros informados."
        lines = [f"Encontrei {len(rows)} registro(s):"]
        for row in rows[:20]:
            visible = []
            for k, v in row.items():
                if v is None or str(v).strip() == "":
                    continue
                text = str(v).replace("\n", " ")
                if len(text) > 160:
                    text = text[:157] + "..."
                visible.append(f"{k}: {text}")
            lines.append("- " + " | ".join(visible[:6]))
        return "\n".join(lines)

    def _ok(self, plan: GenericQueryPlan, answer: str, data: dict[str, Any], sql: str, params: list[Any]) -> dict[str, Any]:
        return {
            "fallback_to_rag": False,
            "route": "generic_sql_controlled",
            "query_type": plan.intent,
            "answer_text": answer,
            "summary": answer,
            "data": data,
            "plan": plan.__dict__,
            "technical": {"sql": sql, "params": params, "controlled_sql": True},
            "sources": {"deterministic": True, "engine": "duckdb", "generic_engine": True},
        }

    def _error(self, message: str, plan: GenericQueryPlan) -> dict[str, Any]:
        return {
            "fallback_to_rag": False,
            "route": "generic_sql_error",
            "query_type": plan.intent,
            "answer_text": message,
            "summary": message,
            "plan": plan.__dict__,
            "sources": {"deterministic": True, "engine": "duckdb", "generic_engine": True},
        }

    def _save_context(self, case_id: str, plan: GenericQueryPlan, result: dict[str, Any] | None) -> None:
        try:
            mem = dict(self.store.memory.get(case_id) or {})
            mem["generic_last_plan"] = plan.__dict__
            if result:
                mem["generic_last_result"] = result.get("data") or {}
            self.store.memory[case_id] = mem
        except Exception:
            pass

    def _schema_public(self, schema: dict[str, AgentSchemaField]) -> dict[str, Any]:
        return {k: {"source": v.source, "type": v.type, "aliases": v.aliases[:6]} for k, v in list(schema.items())[:80]}
