from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class TableRef:
    source_path: str
    filename: str
    sheet_name: str
    row_count: int
    columns: list[str]


class TabularQueryService:
    """Camada tabular/analítica genérica do Nexus.

    Objetivo:
    - Manter o comportamento bom da V1 para arquivos anexados.
    - Permitir que a base Gabbi sincronizada como CSV seja consultada como tabela.
    - Suportar follow-up contextual: "detalhe eles", "liste cada uma", "dessas CHGs".
    - Ser agnóstico de agente/domínio: não assume VIVO, SAP, CHG ou INC como regra fixa;
      apenas reconhece padrões comuns de códigos e colunas quando existirem.
    """

    QUANT_MARKERS = [
        "quantos", "quantas", "quantidade", "número", "numero", "total", "conta", "contar",
        "listar", "liste", "quais", "mostre", "me dê", "me de", "agrupe", "agrupa",
        "distribuição", "distribuicao", "média", "media", "soma", "top", "maiores", "menores",
        "incidentes", "registros", "linhas", "chamados", "changes", "change", "detalhe", "descreva",
    ]

    FOLLOWUP_MARKERS = [
        "eles", "elas", "deles", "delas", "destes", "destas", "desses", "dessas", "estes", "estas",
        "cada um", "cada uma", "os registros", "as linhas", "esses registros", "essas linhas",
        "me descreva", "descreva", "detalhe", "detalhar", "listar eles", "liste eles", "liste elas",
    ]

    DETAIL_MARKERS = ["descreva", "detalhe", "detalhar", "cada uma", "cada um", "explique", "fale sobre"]

    # Aliases genéricos. O serviço resolve somente se a coluna existir no CSV/tabela.
    COLUMN_ALIASES = {
        "codigo_tipo": ["codigo_tipo", "código tipo", "tipo código", "tipo do código", "tipo_registro", "record_type"],
        "codigo_principal": ["codigo_principal", "código principal", "codigo", "código", "identificador", "id principal"],
        "numero": ["numero", "número", "num", "registro", "ticket", "chamado", "change", "incidente", "id"],
        "prioridade": ["prioridade", "nível", "nivel", "criticidade", "p"],
        "canal": ["canal", "canal impactado", "origem", "origem/canal"],
        "causa": ["causa", "motivo", "origem do problema", "tipo de causa", "causa origem", "causa_origem"],
        "incidente": ["incidente", "ticket", "chamado", "id", "número do incidente", "numero do incidente"],
        "data": ["data", "data abertura", "abertura", "criado em", "data de criação", "data criação", "created_on"],
        "mes": ["mes", "mês", "competencia", "competência", "month", "ano_mes"],
        "status": ["status", "situação", "situacao", "estado"],
        "estado": ["estado", "state", "situação", "situacao"],
        "severidade": ["severidade", "impacto", "urgência", "urgencia"],
        "mesa": ["mesa", "squad", "time", "equipe", "responsável", "responsavel", "grupo_atribuicao", "grupo de atribuição"],
        "categoria": ["categoria", "tipo", "assunto", "tema", "category"],
        "ic_impactado": ["ic_impactado", "ic impactado", "item impactado", "serviço", "servico", "ci", "configuration item"],
        "article_text": ["article_text", "texto", "conteudo", "conteúdo", "descricao", "descrição", "resumo"],
        "topic_name": ["topic_name", "tópico", "topico", "nome do tópico", "nome"],
    }

    CODE_RX = re.compile(r"\b([A-Z]{2,8}\d{2,12}|[A-Z]{2,8}[-_]?\d{2,12}|ID:\d+(?:\.\d+)?)\b", re.IGNORECASE)
    TYPE_TOKEN_RX = re.compile(r"\b(CHG|INC|REQ|TASK|PRB|Z[A-Z]{1,6})\b", re.IGNORECASE)

    MONTHS = {
        "janeiro": "01", "jan": "01",
        "fevereiro": "02", "fev": "02",
        "março": "03", "marco": "03", "mar": "03",
        "abril": "04", "abr": "04",
        "maio": "05", "mai": "05",
        "junho": "06", "jun": "06",
        "julho": "07", "jul": "07",
        "agosto": "08", "ago": "08",
        "setembro": "09", "set": "09",
        "outubro": "10", "out": "10",
        "novembro": "11", "nov": "11",
        "dezembro": "12", "dez": "12",
    }

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._catalog_cache: dict[str, list[TableRef]] = {}
        self._last_plan_cache: dict[str, dict[str, Any]] = {}
        self.detail_limit = int(os.getenv("GABBI_TABULAR_DETAIL_LIMIT", "150"))

    def build_catalog(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        tables = self._load_tables(documents)
        self._catalog_cache[case_id] = tables
        return {
            "tables": [
                {
                    "filename": t.filename,
                    "sheet_name": t.sheet_name,
                    "row_count": t.row_count,
                    "columns": t.columns,
                }
                for t in tables
            ],
            "tables_count": len(tables),
        }

    def answer_question(self, case_id: str, question: str, documents: list[dict[str, Any]], mode: str = "executive") -> dict[str, Any] | None:
        tables = self._catalog_cache.get(case_id) or self._load_tables(documents)
        self._catalog_cache[case_id] = tables
        if not tables:
            return None
        if not self._looks_tabular(question, tables):
            return None

        last_context = self._last_plan_cache.get(case_id)
        plan = None
        if self._is_contextual_followup(question) and last_context:
            plan = self._plan_followup(question, last_context, tables)

        if not plan:
            plan = self._plan_question(question, tables)

        if not plan:
            return None

        execution = self._execute_plan(plan, tables)

        # Se a consulta de follow-up voltou 0, tenta preservar a base anterior sem trocar coluna indevidamente.
        if not execution.get("success") or (execution.get("rows_filtered") == 0 and self._is_contextual_followup(question) and last_context):
            fallback_plan = self._plan_followup(question, last_context, tables, strict=True)
            if fallback_plan and fallback_plan != plan:
                fallback_execution = self._execute_plan(fallback_plan, tables)
                if fallback_execution.get("success") and fallback_execution.get("rows_filtered", 0) > 0:
                    plan, execution = fallback_plan, fallback_execution

        if not execution.get("success"):
            return {
                "route": "tabular",
                "query_type": plan.get("intent"),
                "answer_text": execution.get("message"),
                "summary": execution.get("message"),
                "technical": {"plan": plan, "execution": execution, "last_context_used": bool(last_context)},
                "evidences": [],
                "evidence_files": [],
            }

        answer = self._format_answer(question, plan, execution, mode)
        payload = {
            "route": "tabular",
            "query_type": plan.get("intent"),
            "answer_text": answer,
            "summary": answer,
            "technical": {"plan": plan, "execution": execution, "last_context_used": bool(last_context and plan.get("followup"))},
            "evidences": execution.get("evidences", []),
            "evidence_files": execution.get("evidence_files", []),
        }

        self._remember_last_plan(case_id, question, plan, execution)
        return payload

    def _remember_last_plan(self, case_id: str, question: str, plan: dict[str, Any], execution: dict[str, Any]) -> None:
        # Mantém filtros de count/group/list para follow-ups como "detalhe cada uma delas".
        if execution.get("success"):
            self._last_plan_cache[case_id] = {
                "question": question,
                "plan": plan,
                "execution": {
                    "type": execution.get("type"),
                    "rows_considered": execution.get("rows_considered"),
                    "rows_filtered": execution.get("rows_filtered"),
                    "count": execution.get("count"),
                    "table": execution.get("table"),
                    "filters": execution.get("filters", []),
                    "group_by": execution.get("group_by"),
                },
                "remembered_at": pd.Timestamp.utcnow().isoformat(),
            }

    def _looks_tabular(self, question: str, tables: list[TableRef]) -> bool:
        q = question.lower()
        if self._is_contextual_followup(question) and self._last_plan_cache:
            return True
        if any(marker in q for marker in self.QUANT_MARKERS):
            return True
        if self.CODE_RX.search(question):
            return True
        if re.search(r"\bp\d+\b", q):
            return True
        joined_cols = " ".join(" ".join(t.columns).lower() for t in tables)
        return SequenceMatcher(None, q, joined_cols).ratio() > 0.12

    def _is_contextual_followup(self, question: str) -> bool:
        q = question.lower().strip()
        return any(marker in q for marker in self.FOLLOWUP_MARKERS)

    def _plan_question(self, question: str, tables: list[TableRef]) -> dict[str, Any] | None:
        # LLM pode ajudar, mas a heurística deterministicamente evita trocar codigo_tipo por tipo.
        planned = None
        if self.llm_service and self.llm_service.status().get("enabled"):
            planned = self._plan_with_llm(question, tables)
        heuristic = self._plan_with_heuristics(question, tables)
        if not planned:
            return heuristic
        if heuristic:
            # Mescla filtros determinísticos críticos (códigos, tipo, mês) ao plano do LLM.
            planned_filters = planned.setdefault("filters", [])
            for hf in heuristic.get("filters", []):
                if not any(f.get("column") == hf.get("column") and str(f.get("value")).lower() == str(hf.get("value")).lower() for f in planned_filters):
                    planned_filters.append(hf)
            planned.setdefault("target_filename", heuristic.get("target_filename"))
            planned.setdefault("target_sheet", heuristic.get("target_sheet"))
            if heuristic.get("intent") in {"count", "group", "list"}:
                planned["intent"] = heuristic.get("intent")
        return planned

    def _plan_followup(self, question: str, last_context: dict[str, Any], tables: list[TableRef], strict: bool = False) -> dict[str, Any] | None:
        last_plan = dict(last_context.get("plan") or {})
        last_execution = last_context.get("execution") or {}
        if not last_plan:
            return None

        q = question.lower()
        intent = "list"
        if any(x in q for x in ["quantos", "quantidade", "total", "contar"]):
            intent = "count"
        elif " por " in q and any(x in q for x in ["agrupe", "agrupa", "distribuição", "distribuicao", "total", "quantidade"]):
            intent = "group"
        elif any(x in q for x in self.DETAIL_MARKERS):
            intent = "list"

        plan = {
            "use_tabular": True,
            "intent": intent,
            "target_filename": last_plan.get("target_filename") or (last_execution.get("table") or {}).get("filename"),
            "target_sheet": last_plan.get("target_sheet") or (last_execution.get("table") or {}).get("sheet_name"),
            "filters": list(last_plan.get("filters") or last_execution.get("filters") or []),
            "group_by": last_plan.get("group_by"),
            "limit": self.detail_limit if intent == "list" else last_plan.get("limit", 20),
            "answer_style": "markdown",
            "followup": True,
            "followup_from_question": last_context.get("question"),
        }

        if intent == "group":
            target = self._find_table(plan, tables)
            raw = q.split(" por ", 1)[1].strip() if " por " in q else ""
            raw = re.split(r"[\?\.,]", raw)[0].strip()
            if target:
                plan["group_by"] = self._resolve_column(target.columns, [raw]) or self._resolve_column(target.columns, [raw.split()[0]])

        # Se o follow-up traz um código novo explícito, adiciona/atualiza filtro de código sem perder filtros úteis.
        if not strict:
            target = self._find_table(plan, tables)
            if target:
                for f in self._extract_code_filters(question, target.columns):
                    plan["filters"] = [old for old in plan["filters"] if old.get("column") != f.get("column")]
                    plan["filters"].append(f)

        return plan

    def _plan_with_llm(self, question: str, tables: list[TableRef]) -> dict[str, Any] | None:
        catalog = []
        for t in tables:
            catalog.append({
                "filename": t.filename,
                "sheet_name": t.sheet_name,
                "row_count": t.row_count,
                "columns": t.columns,
            })
        system_prompt = (
            "Você é um planejador de consultas tabulares do GABBI. "
            "Receba a pergunta do usuário e o catálogo das tabelas disponíveis. "
            "Decida se a pergunta deve ser respondida com consulta estruturada. "
            "Retorne apenas JSON com as chaves: use_tabular(boolean), intent(count|list|group), "
            "target_filename, target_sheet, filters(array), group_by(string|null), limit(number|null), answer_style. "
            "Cada filtro deve ter: column, operator(eq|contains|gte|lte|between), value. "
            "Use apenas nomes de colunas que existam no catálogo. "
            "Para códigos como CHG/INC/REQ/ZAA, prefira colunas codigo_tipo, codigo_principal ou numero quando existirem; evite a coluna tipo se codigo_tipo existir."
        )
        user_prompt = f"Pergunta: {question}\n\nCatálogo: {catalog}"
        payload = self.llm_service.generate_json(system_prompt, user_prompt)
        if not payload or not payload.get("use_tabular"):
            return None
        if payload.get("intent") not in {"count", "list", "group"}:
            payload["intent"] = "list"
        return payload

    def _plan_with_heuristics(self, question: str, tables: list[TableRef]) -> dict[str, Any] | None:
        q = question.lower()
        intent = "list"
        if any(x in q for x in ["quantos", "quantas", "quantidade", "número", "numero", "total", "contar"]):
            intent = "count"
        elif " por " in q and any(x in q for x in ["quantos", "quantidade", "total", "distribuição", "distribuicao", "agrup"]):
            intent = "group"
        elif any(x in q for x in ["listar", "liste", "quais", "mostre", "descreva", "detalhe"]):
            intent = "list"

        target = self._pick_best_table(q, tables)
        if not target:
            return None
        filters: list[dict[str, Any]] = []

        filters.extend(self._extract_code_filters(question, target.columns))

        p_match = re.search(r"\b(p\d+)\b", q)
        if p_match:
            col = self._resolve_column(target.columns, ["prioridade", "severidade"])
            if col:
                filters.append({"column": col, "operator": "contains", "value": p_match.group(1).upper()})

        month_terms = self._extract_month_terms(q)
        month_col = self._resolve_column(target.columns, ["mes"])
        date_col = self._resolve_column(target.columns, ["data"])
        if month_terms:
            if month_col:
                filters.append({"column": month_col, "operator": "eq", "value": month_terms[0]})
            elif date_col:
                filters.append({"column": date_col, "operator": "contains", "value": month_terms[0]})

        # Termos de domínio livres: ecomm, app, aura etc. Aplica no melhor campo disponível.
        free_terms = self._extract_free_business_terms(q)
        if free_terms:
            candidate_cols = [
                self._resolve_column(target.columns, ["categoria"]),
                self._resolve_column(target.columns, ["canal"]),
                self._resolve_column(target.columns, ["ic_impactado"]),
                self._resolve_column(target.columns, ["topic_name"]),
                self._resolve_column(target.columns, ["article_text"]),
            ]
            candidate_cols = [c for c in candidate_cols if c]
            for term in free_terms:
                # OR não existe no plano simples; escolhe a coluna que mais casa com dados via amostra dinâmica na execução.
                if candidate_cols:
                    filters.append({"column": candidate_cols[0], "operator": "contains", "value": term})

        group_by = None
        if intent == "group":
            raw = q.split(" por ", 1)[1].strip() if " por " in q else ""
            raw = re.split(r"[\?\.,]", raw)[0].strip()
            group_by = self._resolve_column(target.columns, [raw]) or self._resolve_column(target.columns, [raw.split()[0]])

        return {
            "use_tabular": True,
            "intent": intent,
            "target_filename": target.filename,
            "target_sheet": target.sheet_name,
            "filters": self._dedupe_filters(filters),
            "group_by": group_by,
            "limit": self.detail_limit if intent == "list" and any(x in q for x in self.DETAIL_MARKERS) else 20,
            "answer_style": "markdown",
        }

    def _extract_code_filters(self, question: str, columns: list[str]) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        q = question.upper()
        exact_codes = [m.group(1).upper().replace("_", "-") for m in self.CODE_RX.finditer(q)]
        for code in exact_codes:
            # Código completo: usar numero/codigo_principal/article_text como fallback.
            if code.startswith("ID:"):
                col = self._resolve_column(columns, ["codigo_principal", "numero", "article_text"])
            else:
                col = self._resolve_column(columns, ["numero", "codigo_principal", "article_text"])
            if col:
                filters.append({"column": col, "operator": "contains", "value": code})

        # Tipo sem número: CHG, INC, REQ. Preferir codigo_tipo. Não usar "tipo" se codigo_tipo existir.
        type_tokens = [m.group(1).upper() for m in self.TYPE_TOKEN_RX.finditer(q)]
        for token in type_tokens:
            if any(code.startswith(token) and len(code) > len(token) for code in exact_codes):
                continue
            col = self._resolve_column(columns, ["codigo_tipo"])
            if not col:
                col = self._resolve_column(columns, ["codigo_principal", "numero", "tipo", "article_text"])
            if col:
                filters.append({"column": col, "operator": "contains", "value": token})
        return self._dedupe_filters(filters)

    def _extract_free_business_terms(self, question_lower: str) -> list[str]:
        # Stopwords básicas para não filtrar por termos vagos.
        stop = {
            "me", "liste", "listar", "total", "quantos", "quantas", "temos", "de", "do", "da", "dos", "das",
            "em", "no", "na", "por", "para", "com", "os", "as", "o", "a", "incidentes", "changes", "change",
            "registros", "chamados", "categoria", "categorias", "entao", "então", "descreva", "detalhe",
        }
        candidates = []
        for token in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", question_lower):
            if token in stop:
                continue
            if re.match(r"^20\d{2}$", token) or re.match(r"^\d{1,2}$", token):
                continue
            if self.TYPE_TOKEN_RX.fullmatch(token.upper()):
                continue
            candidates.append(token.upper() if token.isupper() else token)
        return candidates[:2]

    def _dedupe_filters(self, filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        seen = set()
        for f in filters:
            key = (f.get("column"), f.get("operator"), str(f.get("value")).lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def _pick_best_table(self, question_lower: str, tables: list[TableRef]) -> TableRef | None:
        # Prioriza tabela Gabbi ativa quando existir, mas continua genérico para uploads.
        best = None
        best_score = -1.0
        for table in tables:
            hay = f"{table.filename} {table.sheet_name} {' '.join(table.columns)}".lower()
            score = SequenceMatcher(None, question_lower, hay).ratio()
            score += sum(0.08 for token in question_lower.split() if token in hay)
            if "gabbi_knowledge_table_active" in table.filename.lower():
                score += 0.25
            if re.search(r"\bp\d+\b", question_lower) and self._resolve_column(table.columns, ["prioridade", "severidade"]):
                score += 0.4
            if self.TYPE_TOKEN_RX.search(question_lower) and self._resolve_column(table.columns, ["codigo_tipo", "numero"]):
                score += 0.4
            if score > best_score:
                best, best_score = table, score
        return best

    def _find_table(self, plan: dict[str, Any], tables: list[TableRef]) -> TableRef | None:
        for table in tables:
            if table.filename == plan.get("target_filename") and table.sheet_name == plan.get("target_sheet"):
                return table
        return self._pick_best_table("", tables)

    def _execute_plan(self, plan: dict[str, Any], tables: list[TableRef]) -> dict[str, Any]:
        target = self._find_table(plan, tables)
        if not target:
            return {"success": False, "message": "Não encontrei a aba/tabela selecionada para executar a consulta."}

        df = self._load_dataframe(target)
        if df.empty:
            return {"success": False, "message": "A tabela está vazia."}

        filtered = df.copy()
        applied_filters = []
        for filt in plan.get("filters", []):
            column = filt.get("column")
            if column not in filtered.columns:
                continue
            op = filt.get("operator", "contains")
            value = filt.get("value")
            if value is None or value == "":
                continue
            series = filtered[column].astype(str).fillna("")
            before = len(filtered)
            if op == "eq":
                filtered = filtered[series.str.lower() == str(value).lower()]
            elif op == "contains":
                filtered = filtered[series.str.contains(str(value), case=False, na=False, regex=False)]
            elif op == "gte":
                filtered = filtered[series >= str(value)]
            elif op == "lte":
                filtered = filtered[series <= str(value)]
            elif op == "between" and isinstance(value, list) and len(value) == 2:
                filtered = filtered[(series >= str(value[0])) & (series <= str(value[1]))]
            applied_filters.append({"column": column, "operator": op, "value": value, "before": before, "after": len(filtered)})

        intent = plan.get("intent", "list")
        evidences = [{
            "filename": target.filename,
            "sheet_name": target.sheet_name,
            "score": 1.0,
            "excerpt": f"Tabela {target.sheet_name} com {target.row_count} linhas e {len(target.columns)} colunas.",
        }]
        if intent == "count":
            return {
                "success": True,
                "type": "count",
                "count": int(len(filtered)),
                "rows_considered": int(len(df)),
                "rows_filtered": int(len(filtered)),
                "table": {"filename": target.filename, "sheet_name": target.sheet_name},
                "filters": applied_filters,
                "preview": filtered.head(10).fillna("").to_dict(orient="records"),
                "evidences": evidences,
                "evidence_files": [target.filename],
            }
        if intent == "group":
            group_by = plan.get("group_by")
            if not group_by or group_by not in filtered.columns:
                return {"success": False, "message": "Não consegui identificar a coluna para agrupar a consulta."}
            grouped = filtered[group_by].astype(str).fillna("").value_counts(dropna=False).reset_index()
            grouped.columns = [group_by, "total"]
            return {
                "success": True,
                "type": "group",
                "rows_considered": int(len(df)),
                "rows_filtered": int(len(filtered)),
                "table": {"filename": target.filename, "sheet_name": target.sheet_name},
                "filters": applied_filters,
                "group_by": group_by,
                "results": grouped.head(100).to_dict(orient="records"),
                "evidences": evidences,
                "evidence_files": [target.filename],
            }
        limit = int(plan.get("limit") or 20)
        return {
            "success": True,
            "type": "list",
            "rows_considered": int(len(df)),
            "rows_filtered": int(len(filtered)),
            "table": {"filename": target.filename, "sheet_name": target.sheet_name},
            "filters": applied_filters,
            "results": filtered.head(limit).fillna("").to_dict(orient="records"),
            "columns": list(filtered.columns),
            "returned_rows": int(min(len(filtered), limit)),
            "limit": limit,
            "evidences": evidences,
            "evidence_files": [target.filename],
        }

    def _format_answer(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str:
        if self.llm_service and self.llm_service.status().get("enabled"):
            answer = self._format_with_llm(question, plan, execution, mode)
            if answer:
                return answer
        table = execution["table"]
        header = f"## Resposta\n\nConsulta executada na tabela **{table['filename']} / {table['sheet_name']}**."
        filters_md = self._filters_to_markdown(execution.get("filters", []))
        followup_note = "\n\n> Consulta contextual: foram reaproveitados os filtros da pergunta anterior." if plan.get("followup") else ""
        if execution["type"] == "count":
            return (
                f"{header}{followup_note}\n\n"
                f"**Total encontrado:** {execution['count']} registros.\n\n"
                f"**Linhas consideradas:** {execution['rows_considered']}\n\n"
                f"### Filtros aplicados\n{filters_md}\n\n"
                f"### Amostra\n{self._records_to_markdown(execution.get('preview', []))}"
            )
        if execution["type"] == "group":
            return (
                f"{header}{followup_note}\n\n"
                f"### Distribuição por **{execution['group_by']}**\n\n"
                f"{self._records_to_markdown(execution.get('results', []))}\n\n"
                f"### Filtros aplicados\n{filters_md}"
            )
        total = execution.get("rows_filtered", 0)
        returned = execution.get("returned_rows", 0)
        return (
            f"{header}{followup_note}\n\n"
            f"**Registros encontrados:** {total}\n"
            f"**Registros exibidos nesta resposta:** {returned}\n\n"
            f"### Filtros aplicados\n{filters_md}\n\n"
            f"### Registros\n{self._records_to_markdown(execution.get('results', []))}"
        )

    def _format_with_llm(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str | None:
        system_prompt = (
            "Você é um analista sênior do GABBI. Receba o resultado de uma consulta tabular já executada e redija a resposta em markdown bem formatado. "
            "Nunca invente números. Use exatamente os resultados recebidos. "
            "Se o plano tiver followup=true, deixe claro que os filtros da pergunta anterior foram reaproveitados. "
            "Sempre inclua: resposta direta, cobertura da consulta, filtros aplicados e observações úteis. "
            "Se houver muitos registros, descreva os principais campos de cada linha retornada e informe a quantidade total encontrada."
        )
        if mode == "executive":
            system_prompt += " Use linguagem executiva e simples."
        elif mode == "technical":
            system_prompt += " Use linguagem técnica e inclua detalhes da execução."
        # Evita prompt gigante: manda até o limite já aplicado em execution.
        user_prompt = f"Pergunta: {question}\n\nPlano: {plan}\n\nResultado executado: {execution}"
        return self.llm_service.generate_chat(system_prompt, [], user_prompt, temperature=0)

    def _filters_to_markdown(self, filters: list[dict[str, Any]]) -> str:
        if not filters:
            return "- Nenhum filtro específico foi aplicado."
        lines = []
        for f in filters:
            after = f" — {f.get('after')} resultado(s) após o filtro" if "after" in f else ""
            lines.append(f"- **{f['column']}** {f['operator']} `{f['value']}`{after}")
        return "\n".join(lines)

    def _records_to_markdown(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "Nenhum registro encontrado."
        preferred = [
            "numero", "codigo_principal", "codigo_tipo", "mes", "tipo", "estado", "status", "prioridade",
            "categoria", "canal", "ic_impactado", "grupo_atribuicao", "data_inicio_planejada",
            "data_termino_planejada", "topic_name", "article_text",
        ]
        available = list(records[0].keys())
        columns = [c for c in preferred if c in available]
        if not columns:
            columns = available[:8]
        else:
            # Completa com algumas colunas relevantes não previstas.
            for c in available:
                if c not in columns and len(columns) < 10:
                    columns.append(c)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in records[:150]:
            vals = [str(row.get(col, "")).replace("\n", " ").replace("|", " /")[:180] for col in columns]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _load_tables(self, documents: list[dict[str, Any]]) -> list[TableRef]:
        tables: list[TableRef] = []
        for doc in documents:
            path = Path(doc.get("path", ""))
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            try:
                if suffix == ".csv":
                    df = pd.read_csv(path, dtype=str, keep_default_na=False)
                    tables.append(TableRef(str(path), path.name, "csv", int(df.shape[0]), [str(c).strip() for c in df.columns.tolist()]))
                elif suffix in {".xlsx", ".xlsm", ".xls"}:
                    xl = pd.ExcelFile(path)
                    for sheet in xl.sheet_names:
                        df = xl.parse(sheet, dtype=str).fillna("")
                        tables.append(TableRef(str(path), path.name, sheet, int(df.shape[0]), [str(c).strip() for c in df.columns.tolist()]))
            except Exception:
                continue
        return tables

    def _load_dataframe(self, table: TableRef) -> pd.DataFrame:
        path = Path(table.source_path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:
            df = pd.read_excel(path, sheet_name=table.sheet_name, dtype=str).fillna("")
        df.columns = [str(c).strip() for c in df.columns]
        return df.fillna("")

    def _resolve_column(self, columns: list[str], semantic_candidates: list[str]) -> str | None:
        # Preferência explícita: se busca codigo_tipo e a coluna existe, retorna direto.
        lower_map = {c.lower(): c for c in columns}
        for candidate in semantic_candidates:
            candidate_norm = self._norm(candidate).replace(" ", "_")
            if candidate_norm in lower_map:
                return lower_map[candidate_norm]
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]

        normalized = {col: self._norm(col) for col in columns}
        best_col = None
        best_score = 0.0
        expanded_terms: list[str] = []
        for candidate in semantic_candidates:
            expanded_terms.extend(self.COLUMN_ALIASES.get(candidate, [candidate]))
            expanded_terms.append(candidate)
        for col, norm_col in normalized.items():
            for term in expanded_terms:
                norm_term = self._norm(term)
                score = SequenceMatcher(None, norm_col, norm_term).ratio()
                if norm_term in norm_col or norm_col in norm_term:
                    score += 0.4
                # Desempate para evitar escolher "tipo" quando existe "codigo_tipo".
                if "codigo tipo" in norm_col and norm_term in {"codigo tipo", "tipo codigo", "record type"}:
                    score += 0.5
                if score > best_score:
                    best_col, best_score = col, score
        return best_col if best_score >= 0.42 else None

    def _extract_month_terms(self, question_lower: str) -> list[str]:
        found: list[str] = []
        # 2025-12 / 2025/12
        for m in re.finditer(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", question_lower):
            found.append(f"{m.group(1)}-{int(m.group(2)):02d}")
        # 12 de 2025
        for m in re.finditer(r"\b(0?[1-9]|1[0-2])\s*(?:de|/)\s*(20\d{2})\b", question_lower):
            found.append(f"{m.group(2)}-{int(m.group(1)):02d}")
        # dezembro de 2025
        for name, month in self.MONTHS.items():
            if name in question_lower:
                y = re.search(r"\b(20\d{2})\b", question_lower)
                if y:
                    found.append(f"{y.group(1)}-{month}")
                else:
                    found.append(month)
        # Dedup
        out = []
        for item in found:
            if item not in out:
                out.append(item)
        return out

    def _norm(self, value: str) -> str:
        lowered = value.lower().strip()
        replacements = {
            "ç": "c", "ã": "a", "á": "a", "à": "a", "â": "a", "ä": "a",
            "é": "e", "ê": "e", "è": "e", "í": "i", "ì": "i",
            "ó": "o", "ô": "o", "õ": "o", "ò": "o", "ú": "u", "ù": "u",
        }
        for src, dst in replacements.items():
            lowered = lowered.replace(src, dst)
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return " ".join(lowered.split())
