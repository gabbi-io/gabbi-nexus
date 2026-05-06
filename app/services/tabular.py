from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.query_intelligence import QueryIntelligence


@dataclass
class TableRef:
    source_path: str
    filename: str
    sheet_name: str
    row_count: int
    columns: list[str]


class TabularQueryService:
    """Ferramenta tabular auxiliar.

    Importante: este serviço NÃO deve dominar a conversa. Ele só responde quando a
    QueryIntelligence indicar que a pergunta é analítica/listagem/filtro estruturado.
    Perguntas abertas devem voltar para o RAG/Nexus via graph.py.
    """

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._catalog_cache: dict[str, list[TableRef]] = {}
        self._conversation_context: dict[str, dict[str, Any]] = {}
        self.query_ai = QueryIntelligence()

    def build_catalog(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        tables = self._load_tables(documents)
        self._catalog_cache[case_id] = tables
        # Novo case/conversa deve começar sem filtros temporários.
        self._conversation_context.pop(case_id, None)
        return {
            "tables": [
                {"filename": t.filename, "sheet_name": t.sheet_name, "row_count": t.row_count, "columns": t.columns}
                for t in tables
            ],
            "tables_count": len(tables),
        }

    def clear_context(self, case_id: str) -> None:
        self._conversation_context.pop(case_id, None)

    def answer_question(self, case_id: str, question: str, documents: list[dict[str, Any]], mode: str = "executive") -> dict[str, Any] | None:
        tables = self._catalog_cache.get(case_id) or self._load_tables(documents)
        self._catalog_cache[case_id] = tables
        if not tables:
            return None

        target = self._pick_table(tables)
        if not target:
            return None

        previous_context = self._conversation_context.get(case_id) or {}
        plan = self.query_ai.plan(question, target.columns, previous_context)
        if not plan.get("use_tabular"):
            # Pergunta aberta ou mudança de assunto: limpa filtros temporários para não contaminar.
            if plan.get("reason") in {"reset_or_general_context_question", "no_tabular_signal", "open_question_without_filters"}:
                self.clear_context(case_id)
            return None

        plan["target_filename"] = target.filename
        plan["target_sheet"] = target.sheet_name

        execution = self._execute_plan(plan, target)
        if not execution.get("success"):
            return None

        # Se a consulta ficou vazia e a confiança não é alta, deixa o graph.py tentar RAG limpo.
        if execution.get("rows_filtered", 0) == 0 and float(plan.get("confidence", 0)) < 0.78:
            return None

        answer = self._format_answer(question, plan, execution, mode)
        result = {
            "route": "tabular",
            "query_type": plan.get("intent"),
            "answer_text": answer,
            "summary": answer,
            "technical": {"plan": plan, "execution": execution},
            "evidences": execution.get("evidences", []),
            "evidence_files": execution.get("evidence_files", []),
        }

        self._remember_context(case_id, plan, execution)
        return result

    def _pick_table(self, tables: list[TableRef]) -> TableRef | None:
        if not tables:
            return None
        # Prioriza a projeção tabular ativa da base Gabbi, mas preserva anexos CSV/XLSX.
        preferred = [t for t in tables if t.filename.startswith("gabbi_knowledge_table_active_")]
        if preferred:
            return max(preferred, key=lambda t: t.row_count)
        return max(tables, key=lambda t: t.row_count)

    def _remember_context(self, case_id: str, plan: dict[str, Any], execution: dict[str, Any]) -> None:
        # Guarda só contexto de consulta válido. Nunca guarda filtros vazios/ruins.
        filters = execution.get("filters") or plan.get("filters") or []
        if not filters:
            return
        self._conversation_context[case_id] = {
            "filters": filters,
            "intent": plan.get("intent"),
            "table": execution.get("table"),
            "rows_filtered": execution.get("rows_filtered"),
            "last_results_preview": execution.get("preview") or execution.get("results"),
            "updated_turns": 0,
        }

    def _execute_plan(self, plan: dict[str, Any], target: TableRef) -> dict[str, Any]:
        df = self._load_dataframe(target)
        if df.empty:
            return {"success": False, "message": "A tabela está vazia."}

        filtered = df.copy()
        applied_filters = []
        filter_trace = []
        for filt in plan.get("filters", []):
            column = filt.get("column")
            if column not in filtered.columns:
                continue
            op = filt.get("operator", "contains")
            value = filt.get("value")
            if value is None or value == "":
                continue
            before = int(len(filtered))
            series = filtered[column].astype(str).fillna("")
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
            after = int(len(filtered))
            item = {"column": column, "operator": op, "value": value, "before": before, "after": after}
            applied_filters.append({"column": column, "operator": op, "value": value})
            filter_trace.append(item)

        intent = plan.get("intent", "list")
        evidences = [{
            "filename": target.filename,
            "sheet_name": target.sheet_name,
            "score": 1.0,
            "excerpt": f"Tabela {target.sheet_name} com {target.row_count} linhas e {len(target.columns)} colunas.",
        }]

        base = {
            "success": True,
            "rows_considered": int(len(df)),
            "rows_filtered": int(len(filtered)),
            "table": {"filename": target.filename, "sheet_name": target.sheet_name},
            "filters": applied_filters,
            "filter_trace": filter_trace,
            "evidences": evidences,
            "evidence_files": [target.filename],
        }

        if intent == "count":
            return {**base, "type": "count", "count": int(len(filtered)), "preview": filtered.head(8).fillna("").to_dict(orient="records")}

        if intent == "group":
            group_by = plan.get("group_by")
            if not group_by or group_by not in filtered.columns:
                return {"success": False, "message": "Não consegui identificar a coluna para agrupar a consulta."}
            grouped = filtered[group_by].astype(str).fillna("").value_counts(dropna=False).reset_index()
            grouped.columns = [group_by, "total"]
            return {**base, "type": "group", "group_by": group_by, "results": grouped.head(30).to_dict(orient="records")}

        limit = int(plan.get("limit") or 150)
        return {
            **base,
            "type": "list",
            "results": filtered.head(limit).fillna("").to_dict(orient="records"),
            "columns": list(filtered.columns),
            "returned_rows": int(min(len(filtered), limit)),
        }

    def _format_answer(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str:
        if self.llm_service and self.llm_service.status().get("enabled"):
            answer = self._format_with_llm(question, plan, execution, mode)
            if answer:
                return answer
        table = execution["table"]
        header = f"## Resposta\n\nConsulta executada na base **{table['filename']} / {table['sheet_name']}**."
        filters_md = self._filters_to_markdown(execution.get("filter_trace", []))
        if execution["type"] == "count":
            return (
                f"{header}\n\n"
                f"**Total encontrado:** {execution['count']} registros.\n\n"
                f"- Linhas consideradas: {execution['rows_considered']}\n"
                f"- Linhas após filtros: {execution['rows_filtered']}\n\n"
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
            f"**Registros encontrados:** {execution['rows_filtered']}\n"
            f"**Registros retornados nesta resposta:** {execution.get('returned_rows', 0)}\n\n"
            f"### Filtros aplicados\n{filters_md}\n\n"
            f"### Registros\n{self._records_to_markdown(execution.get('results', []))}"
        )

    def _format_with_llm(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str | None:
        # Limita payload para evitar respostas enormes e custo alto.
        safe_execution = dict(execution)
        if isinstance(safe_execution.get("results"), list):
            safe_execution["results"] = safe_execution["results"][:20]
        if isinstance(safe_execution.get("preview"), list):
            safe_execution["preview"] = safe_execution["preview"][:10]
        system_prompt = (
            "Você é um analista sênior do GABBI. Receba o resultado de uma consulta tabular já executada e redija a resposta em markdown. "
            "Nunca invente números. Use exatamente os resultados recebidos. "
            "Não diga que não encontrou se o campo filter_trace mostrar linhas intermediárias relevantes; explique os filtros aplicados. "
            "Se o usuário pediu listagem/detalhamento, descreva os registros retornados com objetividade."
        )
        user_prompt = f"Pergunta: {question}\n\nPlano: {plan}\n\nResultado executado: {safe_execution}"
        return self.llm_service.generate_chat(system_prompt, [], user_prompt, temperature=0)

    def _filters_to_markdown(self, filters: list[dict[str, Any]]) -> str:
        if not filters:
            return "- Nenhum filtro específico foi aplicado."
        lines = []
        for i, f in enumerate(filters, start=1):
            if "before" in f and "after" in f:
                lines.append(f"{i}. `{f['column']}` {f['operator']} `{f['value']}` — {f['before']} → {f['after']} linhas")
            else:
                lines.append(f"{i}. `{f['column']}` {f.get('operator', 'contains')} `{f['value']}`")
        return "\n".join(lines)

    def _records_to_markdown(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "Nenhum registro retornado."
        preferred = ["numero", "codigo_principal", "codigo_tipo", "mes", "tipo", "estado", "status", "prioridade", "grupo_atribuicao", "ic_impactado", "categoria"]
        columns = [c for c in preferred if c in records[0]] or list(records[0].keys())[:8]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in records[:15]:
            vals = [str(row.get(col, "")).replace("\n", " ")[:140] for col in columns]
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
