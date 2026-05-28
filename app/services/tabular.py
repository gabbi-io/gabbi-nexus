from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.query_intelligence import QueryIntelligenceService


@dataclass
class TableRef:
    source_path: str
    filename: str
    sheet_name: str
    row_count: int
    columns: list[str]


class TabularQueryService:
    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._catalog_cache: dict[str, list[TableRef]] = {}
        self._query_context: dict[str, dict[str, Any]] = {}
        self.qi = QueryIntelligenceService()

    def clear_context(self, case_id: str) -> None:
        self._query_context.pop(case_id, None)

    def build_catalog(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        tables = self._load_tables(documents)
        self._catalog_cache[case_id] = tables
        self.clear_context(case_id)
        return {"tables": [{"filename": t.filename, "sheet_name": t.sheet_name, "row_count": t.row_count, "columns": t.columns} for t in tables], "tables_count": len(tables)}

    def answer_question(self, case_id: str, question: str, documents: list[dict[str, Any]], mode: str = "executive", source_preference: str | None = None) -> dict[str, Any] | None:
        tables = self._catalog_cache.get(case_id) or self._load_tables(documents)
        self._catalog_cache[case_id] = tables
        if not tables:
            return None
        target = self._pick_best_table(question, tables)
        if not target:
            return None
        previous = self._query_context.get(case_id)
        plan = self.qi.build_plan(question, target.columns, previous_plan=previous)
        if plan.reset_context:
            self.clear_context(case_id)
            plan = self.qi.build_plan(question, target.columns, previous_plan=None)
        if not plan.use_tabular:
            return None
        execution = self._execute_plan(plan, target)
        plan_dict = self.qi.plan_to_dict(plan)
        if execution.get("success") and execution.get("rows_filtered", 0) == 0:
            inherited_count = sum(1 for f in plan.filters if f.source == "inherited")
            explicit_count = sum(1 for f in plan.filters if f.source == "explicit")
            if inherited_count > 0 or explicit_count == 0:
                message = "Consulta tabular executada, mas nenhum registro foi encontrado com os filtros aplicados."
                return {"route": "tabular_no_match", "query_type": plan.intent, "fallback_to_rag": False, "answer_text": message, "summary": message, "technical": {"plan": plan_dict, "execution": execution}, "evidences": execution.get("evidences", []), "evidence_files": execution.get("evidence_files", []), "sources": {"deterministic": True, "rag_blocked": True}}
        if not execution.get("success"):
            message = execution.get("message", "Não foi possível executar a consulta tabular.")
            return {"route": "tabular_error", "query_type": plan.intent, "fallback_to_rag": False, "answer_text": message, "summary": message, "technical": {"plan": plan_dict, "execution": execution}, "evidences": [], "evidence_files": [], "sources": {"deterministic": True, "rag_blocked": True}}
        self._query_context[case_id] = {**plan_dict, "last_execution_summary": {"rows_considered": execution.get("rows_considered"), "rows_filtered": execution.get("rows_filtered"), "type": execution.get("type")}}
        answer = self._format_answer(question, plan_dict, execution, mode)
        return {"route": "tabular", "query_type": plan.intent, "answer_text": answer, "summary": answer, "technical": {"plan": plan_dict, "execution": execution}, "evidences": execution.get("evidences", []), "evidence_files": execution.get("evidence_files", [])}

    def _pick_best_table(self, question: str, tables: list[TableRef]) -> TableRef | None:
        active = [t for t in tables if "gabbi_knowledge_table_active" in t.filename]
        if active:
            return active[0]
        if len(tables) == 1:
            return tables[0]
        return tables[0] if tables else None

    def _execute_plan(self, plan, table: TableRef) -> dict[str, Any]:
        df = self._load_dataframe(table)
        if df.empty:
            return {"success": False, "message": "A tabela está vazia."}
        filtered = df.copy()
        applied_filters = []
        for filt in plan.filters:
            column = filt.column
            if column not in filtered.columns:
                continue
            value = filt.value
            op = filt.operator or "contains"
            if value is None or str(value).strip() == "":
                continue
            before = int(len(filtered))
            series = filtered[column].astype(str).fillna("")
            if op == "eq":
                filtered = filtered[series.str.lower() == str(value).lower()]
            elif op == "startswith":
                filtered = filtered[series.str.lower().str.startswith(str(value).lower(), na=False)]
            else:
                filtered = filtered[series.str.contains(re.escape(str(value)), case=False, na=False)]
            after = int(len(filtered))
            applied_filters.append({"column": column, "operator": op, "value": value, "source": filt.source, "before": before, "after": after})
        evidences = [{"filename": table.filename, "sheet_name": table.sheet_name, "score": 1.0, "excerpt": f"Tabela {table.sheet_name} com {table.row_count} linhas e {len(table.columns)} colunas."}]
        if plan.intent == "count":
            count_col = self._detect_identifier_column(filtered)
            if count_col:
                count = int(filtered[count_col].astype(str).str.upper().str.strip().replace("", pd.NA).dropna().nunique())
                count_mode = f"distinct:{count_col}"
            else:
                count = int(len(filtered))
                count_mode = "rows"
            return {"success": True, "type": "count", "count": count, "count_mode": count_mode, "rows_considered": int(len(df)), "rows_filtered": int(len(filtered)), "table": {"filename": table.filename, "sheet_name": table.sheet_name}, "filters": applied_filters, "preview": self._safe_records(filtered.head(8)), "evidences": evidences, "evidence_files": [table.filename]}
        if plan.intent == "group":
            group_by = plan.group_by
            if not group_by or group_by not in filtered.columns:
                return {"success": False, "message": "Não consegui identificar a coluna para agrupar a consulta."}
            count_col = self._detect_identifier_column(filtered)
            if count_col:
                grouped = filtered.groupby(group_by, dropna=False)[count_col].nunique().reset_index(name="total")
                grouped = grouped.sort_values(["total", group_by], ascending=[False, True])
            else:
                grouped = filtered[group_by].astype(str).fillna("").value_counts(dropna=False).reset_index()
                grouped.columns = [group_by, "total"]
            return {"success": True, "type": "group", "rows_considered": int(len(df)), "rows_filtered": int(len(filtered)), "table": {"filename": table.filename, "sheet_name": table.sheet_name}, "filters": applied_filters, "group_by": group_by, "results": self._safe_records(grouped.head(50)), "evidences": evidences, "evidence_files": [table.filename]}
        limit = int(plan.limit or 20)
        return {"success": True, "type": "list", "rows_considered": int(len(df)), "rows_filtered": int(len(filtered)), "table": {"filename": table.filename, "sheet_name": table.sheet_name}, "filters": applied_filters, "results": self._safe_records(filtered.head(limit)), "columns": list(filtered.columns), "evidences": evidences, "evidence_files": [table.filename]}

    def _format_answer(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str:
        # Importante: resposta tabular é determinística. Não usar LLM aqui para evitar
        # reinterpretação/síntese de números, especialmente em counts, rankings e agrupamentos.
        table = execution["table"]
        filters_md = self._filters_to_markdown(execution.get("filters", []))
        header = f"## Resposta\n\nConsulta executada na base **{table['filename']} / {table['sheet_name']}**."
        if execution["type"] == "count":
            return f"{header}\n\n**Total encontrado:** {execution['count']} registros.\n\n- Linhas consideradas: {execution['rows_considered']}\n- Linhas após filtros: {execution['rows_filtered']}\n\n### Filtros aplicados\n{filters_md}\n\n### Amostra\n{self._records_to_markdown(execution.get('preview', []))}"
        if execution["type"] == "group":
            return f"{header}\n\n### Distribuição por **{execution['group_by']}**\n\n{self._records_to_markdown(execution.get('results', []))}\n\n### Filtros aplicados\n{filters_md}"
        return f"{header}\n\n**Registros encontrados:** {execution['rows_filtered']}\n\n### Filtros aplicados\n{filters_md}\n\n### Registros\n{self._records_to_markdown(execution.get('results', []))}"

    def _format_with_llm(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str | None:
        system_prompt = "Você é um analista sênior do GABBI. Receba o resultado de uma consulta tabular já executada e redija a resposta em markdown. Nunca invente números. Use exatamente os dados de execution. Explique cobertura, filtros e resultado."
        if mode == "executive":
            system_prompt += " Use linguagem executiva e direta."
        user_prompt = f"Pergunta: {question}\n\nPlano: {plan}\n\nResultado executado: {execution}"
        return self.llm_service.generate_chat(system_prompt, [], user_prompt, temperature=0)

    def _filters_to_markdown(self, filters: list[dict[str, Any]]) -> str:
        if not filters:
            return "- Nenhum filtro específico foi aplicado."
        return "\n".join([f"{i}. `{f.get('column')}` {f.get('operator')} `{f.get('value')}` — {f.get('before')} → {f.get('after')} linhas" for i, f in enumerate(filters, start=1)])

    def _records_to_markdown(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "Nenhum registro encontrado."
        preferred = ["numero", "codigo_principal", "codigo_tipo", "mes", "tipo", "estado", "status", "prioridade", "grupo_atribuicao", "ic_impactado"]
        cols = [c for c in preferred if c in records[0]] or list(records[0].keys())[:8]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in records[:12]:
            vals = [str(row.get(col, "")).replace("\n", " ")[:120] for col in cols]
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

    def _detect_identifier_column(self, df: pd.DataFrame) -> str | None:
        preferred = [
            "numero", "Número", "Numero", "codigo_principal", "Código", "Codigo",
            "number", "id_change", "id_incidente"
        ]
        for col in preferred:
            if col in df.columns:
                return col
        for col in df.columns:
            values = df[col].astype(str).str.upper()
            sample = " ".join(values.head(50).tolist())
            if re.search(r"\b(?:CHG|INC)\d{5,}\b", sample):
                return col
        return None

    def _safe_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        return df.fillna("").astype(str).to_dict(orient="records")
