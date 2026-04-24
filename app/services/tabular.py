from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


@dataclass
class TableRef:
    source_path: str
    filename: str
    sheet_name: str
    row_count: int
    columns: list[str]


class TabularQueryService:
    QUANT_MARKERS = [
        "quantos", "quantas", "quantidade", "número", "numero", "total", "conta", "contar",
        "listar", "liste", "quais", "mostre", "me dê", "me de", "agrupe", "distribuição", "distribuicao",
        "média", "media", "soma", "top", "maiores", "menores", "incidentes", "registros", "linhas",
    ]

    COLUMN_ALIASES = {
        "prioridade": ["prioridade", "nível", "nivel", "criticidade", "p"],
        "canal": ["canal", "canal impactado", "origem", "origem/canal"],
        "causa": ["causa", "motivo", "origem do problema", "tipo de causa"],
        "incidente": ["incidente", "ticket", "chamado", "id", "número do incidente", "numero do incidente"],
        "data": ["data", "data abertura", "abertura", "criado em", "data de criação", "data criação"],
        "status": ["status", "situação", "situacao"],
        "severidade": ["severidade", "impacto", "urgência", "urgencia"],
        "mesa": ["mesa", "squad", "time", "equipe", "responsável", "responsavel"],
        "categoria": ["categoria", "tipo", "assunto", "tema"],
    }

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._catalog_cache: dict[str, list[TableRef]] = {}

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

        plan = self._plan_question(question, tables)
        if not plan:
            return None
        execution = self._execute_plan(plan, tables)
        if not execution.get("success"):
            return {
                "route": "tabular",
                "query_type": plan.get("intent"),
                "answer_text": execution.get("message"),
                "summary": execution.get("message"),
                "technical": {"plan": plan, "execution": execution},
                "evidences": [],
                "evidence_files": [],
            }
        answer = self._format_answer(question, plan, execution, mode)
        return {
            "route": "tabular",
            "query_type": plan.get("intent"),
            "answer_text": answer,
            "summary": answer,
            "technical": {"plan": plan, "execution": execution},
            "evidences": execution.get("evidences", []),
            "evidence_files": execution.get("evidence_files", []),
        }

    def _looks_tabular(self, question: str, tables: list[TableRef]) -> bool:
        q = question.lower()
        if any(marker in q for marker in self.QUANT_MARKERS):
            return True
        if re.search(r"\bp\d+\b", q):
            return True
        joined_cols = " ".join(" ".join(t.columns).lower() for t in tables)
        return SequenceMatcher(None, q, joined_cols).ratio() > 0.12

    def _plan_question(self, question: str, tables: list[TableRef]) -> dict[str, Any] | None:
        if self.llm_service and self.llm_service.status().get("enabled"):
            planned = self._plan_with_llm(question, tables)
            if planned:
                return planned
        return self._plan_with_heuristics(question, tables)

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
            "Retorne apenas JSON com as chaves: use_tabular(boolean), intent(count|list|group), target_filename, target_sheet, filters(array), group_by(string|null), limit(number|null), answer_style. "
            "Cada filtro deve ter: column, operator(eq|contains|gte|lte|between), value. "
            "Use apenas nomes de colunas que existam no catálogo."
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
        elif any(x in q for x in ["listar", "liste", "quais", "mostre"]):
            intent = "list"

        target = self._pick_best_table(q, tables)
        if not target:
            return None
        filters: list[dict[str, Any]] = []

        p_match = re.search(r"\b(p\d+)\b", q)
        if p_match:
            col = self._resolve_column(target.columns, ["prioridade", "severidade"])
            if col:
                filters.append({"column": col, "operator": "contains", "value": p_match.group(1).upper()})

        month_terms = self._extract_month_terms(q)
        date_col = self._resolve_column(target.columns, ["data"])
        if month_terms and date_col:
            filters.append({"column": date_col, "operator": "contains", "value": month_terms[0]})

        for semantic, aliases in self.COLUMN_ALIASES.items():
            if semantic in {"prioridade", "data"}:
                continue
            if any(alias in q for alias in aliases):
                col = self._resolve_column(target.columns, [semantic])
                value = self._extract_filter_value(q, aliases)
                if col and value:
                    filters.append({"column": col, "operator": "contains", "value": value})

        group_by = None
        if intent == "group":
            raw = q.split(" por ", 1)[1].strip()
            raw = re.split(r"[\?\.,]", raw)[0].strip()
            group_by = self._resolve_column(target.columns, [raw]) or self._resolve_column(target.columns, [raw.split()[0]])

        return {
            "use_tabular": True,
            "intent": intent,
            "target_filename": target.filename,
            "target_sheet": target.sheet_name,
            "filters": filters,
            "group_by": group_by,
            "limit": 20,
            "answer_style": "markdown",
        }

    def _pick_best_table(self, question_lower: str, tables: list[TableRef]) -> TableRef | None:
        best = None
        best_score = -1.0
        for table in tables:
            hay = f"{table.filename} {table.sheet_name} {' '.join(table.columns)}".lower()
            score = SequenceMatcher(None, question_lower, hay).ratio()
            score += sum(0.08 for token in question_lower.split() if token in hay)
            if re.search(r"\bp\d+\b", question_lower) and self._resolve_column(table.columns, ["prioridade", "severidade"]):
                score += 0.4
            if score > best_score:
                best, best_score = table, score
        return best

    def _execute_plan(self, plan: dict[str, Any], tables: list[TableRef]) -> dict[str, Any]:
        target = None
        for table in tables:
            if table.filename == plan.get("target_filename") and table.sheet_name == plan.get("target_sheet"):
                target = table
                break
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
            applied_filters.append({"column": column, "operator": op, "value": value})
            series = filtered[column].astype(str).fillna("")
            if op == "eq":
                filtered = filtered[series.str.lower() == str(value).lower()]
            elif op == "contains":
                filtered = filtered[series.str.contains(str(value), case=False, na=False)]
            elif op == "gte":
                filtered = filtered[series >= str(value)]
            elif op == "lte":
                filtered = filtered[series <= str(value)]
            elif op == "between" and isinstance(value, list) and len(value) == 2:
                filtered = filtered[(series >= str(value[0])) & (series <= str(value[1]))]

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
                "preview": filtered.head(5).fillna("").to_dict(orient="records"),
                "evidences": evidences,
                "evidence_files": [target.filename],
            }
        if intent == "group":
            group_by = plan.get("group_by")
            if not group_by or group_by not in filtered.columns:
                return {"success": False, "message": "Não consegui identificar a coluna para agrupar a consulta."}
            grouped = (
                filtered[group_by]
                .astype(str)
                .fillna("")
                .value_counts(dropna=False)
                .reset_index()
            )
            grouped.columns = [group_by, "total"]
            return {
                "success": True,
                "type": "group",
                "rows_considered": int(len(df)),
                "rows_filtered": int(len(filtered)),
                "table": {"filename": target.filename, "sheet_name": target.sheet_name},
                "filters": applied_filters,
                "group_by": group_by,
                "results": grouped.head(20).to_dict(orient="records"),
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
        if execution["type"] == "count":
            return (
                f"{header}\n\n"
                f"**Total encontrado:** {execution['count']} registros.\n\n"
                f"**Linhas consideradas:** {execution['rows_considered']}\n\n"
                f"### Filtros aplicados\n{filters_md}\n\n"
                f"### Amostra\n{self._records_to_markdown(execution.get('preview', []))}"
            )
        if execution["type"] == "group":
            return (
                f"{header}\n\n"
                f"### Distribuição por **{execution['group_by']}**\n\n"
                f"{self._records_to_markdown(execution.get('results', []))}\n\n"
                f"### Filtros aplicados\n{filters_md}"
            )
        return (
            f"{header}\n\n"
            f"**Registros encontrados:** {execution['rows_filtered']}\n\n"
            f"### Filtros aplicados\n{filters_md}\n\n"
            f"### Registros\n{self._records_to_markdown(execution.get('results', []))}"
        )

    def _format_with_llm(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str | None:
        system_prompt = (
            "Você é um analista sênior do GABBI. Receba o resultado de uma consulta tabular já executada e redija a resposta em markdown bem formatado. "
            "Nunca invente números. Use exatamente os resultados recebidos. "
            "Explique de forma humana, objetiva e organizada. "
            "Sempre inclua: resposta direta, cobertura da consulta, filtros aplicados e observações úteis."
        )
        if mode == "executive":
            system_prompt += " Use linguagem executiva e simples."
        elif mode == "technical":
            system_prompt += " Use linguagem técnica e inclua detalhes da execução."
        user_prompt = f"Pergunta: {question}\n\nPlano: {plan}\n\nResultado executado: {execution}"
        return self.llm_service.generate_chat(system_prompt, [], user_prompt, temperature=0)

    def _filters_to_markdown(self, filters: list[dict[str, Any]]) -> str:
        if not filters:
            return "- Nenhum filtro específico foi aplicado."
        return "\n".join([f"- **{f['column']}** {f['operator']} `{f['value']}`" for f in filters])

    def _records_to_markdown(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "Nenhum registro encontrado."
        columns = list(records[0].keys())
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in records[:10]:
            vals = [str(row.get(col, "")).replace("\n", " ")[:120] for col in columns]
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
                    tables.append(TableRef(str(path), path.name, "csv", int(df.shape[0]), [str(c) for c in df.columns.tolist()]))
                elif suffix in {".xlsx", ".xlsm", ".xls"}:
                    xl = pd.ExcelFile(path)
                    for sheet in xl.sheet_names:
                        df = xl.parse(sheet, dtype=str).fillna("")
                        tables.append(TableRef(str(path), path.name, sheet, int(df.shape[0]), [str(c) for c in df.columns.tolist()]))
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
        normalized = {col: self._norm(col) for col in columns}
        best_col = None
        best_score = 0.0
        expanded_terms: list[str] = []
        for candidate in semantic_candidates:
            expanded_terms.extend(self.COLUMN_ALIASES.get(candidate, [candidate]))
            expanded_terms.append(candidate)
        for col, norm_col in normalized.items():
            for term in expanded_terms:
                score = SequenceMatcher(None, norm_col, self._norm(term)).ratio()
                if self._norm(term) in norm_col:
                    score += 0.4
                if score > best_score:
                    best_col, best_score = col, score
        return best_col if best_score >= 0.42 else None

    def _extract_filter_value(self, question_lower: str, aliases: list[str]) -> str | None:
        for alias in aliases:
            pattern = rf"{re.escape(alias)}\s+(?:=|é|e|for|do|da|de|com)?\s*([\w\-_/]+)"
            match = re.search(pattern, question_lower)
            if match:
                return match.group(1)
        return None

    def _extract_month_terms(self, question_lower: str) -> list[str]:
        months = [
            "janeiro", "fevereiro", "março", "marco", "abril", "maio", "junho",
            "julho", "agosto", "setembro", "outubro", "novembro", "dezembro",
        ]
        return [m for m in months if m in question_lower]

    def _norm(self, value: str) -> str:
        lowered = value.lower().strip()
        lowered = lowered.replace("ç", "c").replace("ã", "a").replace("á", "a").replace("à", "a").replace("â", "a")
        lowered = lowered.replace("é", "e").replace("ê", "e").replace("í", "i").replace("ó", "o").replace("ô", "o").replace("õ", "o").replace("ú", "u")
        lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
        return " ".join(lowered.split())
