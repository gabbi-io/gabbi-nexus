from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.query_intelligence import QueryIntelligenceService, PlannedQuery


@dataclass
class TableRef:
    source_path: str
    filename: str
    sheet_name: str
    row_count: int
    columns: list[str]


class TabularQueryService:
    """Consulta tabular genérica para documentos anexados e base Gabbi sincronizada.

    Esta versão evita regras específicas de um agente. Ela usa:
    - catálogo real de colunas;
    - valores existentes na própria tabela;
    - sinônimos genéricos de intenção/códigos;
    - memória tabular curta por case para follow-up.
    """

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._catalog_cache: dict[str, list[TableRef]] = {}
        self._last_context: dict[str, dict[str, Any]] = {}
        self.qi = QueryIntelligenceService()

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

        target = self._pick_best_table(question, tables)
        if not target:
            return None

        df = self._load_dataframe(target)
        if df.empty:
            return None

        last = self._last_context.get(case_id)
        planned = self.qi.build_plan(question, df, last_context=last)
        if not planned.use_tabular:
            return None

        plan = self._planned_to_dict(planned, target)
        execution = self._execute_plan(plan, target, df)

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

        # Salva contexto apenas quando houve consulta útil com filtros confiáveis ou resultado tabular claro.
        self._remember_context(case_id, question, plan, execution)

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

    def _planned_to_dict(self, planned: PlannedQuery, target: TableRef) -> dict[str, Any]:
        return {
            "use_tabular": True,
            "intent": planned.intent,
            "target_filename": target.filename,
            "target_sheet": target.sheet_name,
            "filters": planned.filters,
            "group_by": planned.group_by,
            "limit": planned.limit or 20,
            "followup": planned.followup,
            "confidence": planned.confidence,
            "reason": planned.reason,
            "answer_style": "markdown",
        }

    def _remember_context(self, case_id: str, question: str, plan: dict[str, Any], execution: dict[str, Any]) -> None:
        intent = plan.get("intent")
        if intent == "describe_base":
            # Contexto geral não deve contaminar follow-ups operacionais.
            return
        filters = plan.get("filters") or []
        valid_filters = [f for f in filters if float(f.get("confidence") or 0) >= 0.7 or f.get("source") in {"entity_synonym", "month", "explicit_type", "explicit_code", "code_prefix"}]
        if not valid_filters and intent not in {"count", "group", "list"}:
            return
        self._last_context[case_id] = {
            "question": question,
            "intent": intent,
            "filters": valid_filters,
            "table": execution.get("table"),
            "rows_filtered": execution.get("rows_filtered"),
            "updated_at": pd.Timestamp.utcnow().isoformat(),
        }

    def _pick_best_table(self, question: str, tables: list[TableRef]) -> TableRef | None:
        # Preferir tabela Gabbi ativa quando existir; caso contrário, maior tabela tabular do case.
        gabbi_tables = [t for t in tables if "gabbi_knowledge_table" in t.filename.lower()]
        if gabbi_tables:
            return sorted(gabbi_tables, key=lambda t: t.row_count, reverse=True)[0]
        return sorted(tables, key=lambda t: t.row_count, reverse=True)[0] if tables else None

    def _execute_plan(self, plan: dict[str, Any], target: TableRef, df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {"success": False, "message": "A tabela está vazia."}

        intent = plan.get("intent", "list")
        if intent == "describe_base":
            return self._execute_describe_base(plan, target, df)

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
            before = int(len(filtered))
            series = filtered[column].astype(str).fillna("")
            if op == "eq":
                filtered = filtered[series.str.lower() == str(value).lower()]
            elif op == "contains":
                filtered = filtered[series.str.contains(re.escape(str(value)), case=False, na=False, regex=True)]
            elif op == "gte":
                filtered = filtered[series >= str(value)]
            elif op == "lte":
                filtered = filtered[series <= str(value)]
            elif op == "between" and isinstance(value, list) and len(value) == 2:
                filtered = filtered[(series >= str(value[0])) & (series <= str(value[1]))]
            after = int(len(filtered))
            applied_filters.append({
                "column": column,
                "operator": op,
                "value": value,
                "source": filt.get("source"),
                "confidence": filt.get("confidence"),
                "rows_before": before,
                "rows_after": after,
            })

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
                "preview": filtered.head(8).fillna("").to_dict(orient="records"),
                "evidences": evidences,
                "evidence_files": [target.filename],
            }

        if intent == "group":
            group_by = plan.get("group_by")
            if not group_by or group_by not in filtered.columns:
                return {"success": False, "message": "Não consegui identificar a coluna para agrupar a consulta sem risco de aplicar filtro incorreto."}
            grouped = filtered[group_by].astype(str).fillna("").replace("", "(sem valor)").value_counts(dropna=False).reset_index()
            grouped.columns = [group_by, "total"]
            return {
                "success": True,
                "type": "group",
                "rows_considered": int(len(df)),
                "rows_filtered": int(len(filtered)),
                "table": {"filename": target.filename, "sheet_name": target.sheet_name},
                "filters": applied_filters,
                "group_by": group_by,
                "results": grouped.head(50).to_dict(orient="records"),
                "evidences": evidences,
                "evidence_files": [target.filename],
            }

        limit = int(plan.get("limit") or 20)
        return {
            "success": True,
            "type": "list",
            "rows_considered": int(len(df)),
            "rows_filtered": int(len(filtered)),
            "returned_rows": int(min(len(filtered), limit)),
            "table": {"filename": target.filename, "sheet_name": target.sheet_name},
            "filters": applied_filters,
            "results": filtered.head(limit).fillna("").to_dict(orient="records"),
            "columns": list(filtered.columns),
            "evidences": evidences,
            "evidence_files": [target.filename],
        }

    def _execute_describe_base(self, plan: dict[str, Any], target: TableRef, df: pd.DataFrame) -> dict[str, Any]:
        important_cols = [c for c in df.columns if self.qi.norm(c) in {
            "codigo tipo", "codigo_tipo", "tipo", "categoria", "status", "estado", "prioridade", "mes", "canal", "project_name", "topic_name", "grupo_atribuicao"
        }]
        distributions = {}
        for col in important_cols[:10]:
            vc = df[col].astype(str).fillna("").replace("", "(sem valor)").value_counts(dropna=False).head(10)
            distributions[col] = [{"valor": str(idx), "total": int(val)} for idx, val in vc.items()]
        sample_cols = [c for c in ["project_name", "topic_name", "topic_description", "codigo_tipo", "numero", "tipo", "categoria", "estado", "status", "prioridade"] if c in df.columns]
        if not sample_cols:
            sample_cols = list(df.columns[:8])
        return {
            "success": True,
            "type": "describe_base",
            "rows_considered": int(len(df)),
            "rows_filtered": int(len(df)),
            "table": {"filename": target.filename, "sheet_name": target.sheet_name},
            "filters": [],
            "columns": list(df.columns),
            "distributions": distributions,
            "sample": df[sample_cols].head(8).fillna("").to_dict(orient="records"),
            "evidences": [{"filename": target.filename, "sheet_name": target.sheet_name, "score": 1.0, "excerpt": f"Base tabular com {len(df)} linhas e {len(df.columns)} colunas."}],
            "evidence_files": [target.filename],
        }

    def _format_answer(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str:
        # Formatação determinística primeiro para evitar que o LLM recoloque filtros inválidos.
        deterministic = self._format_deterministic(question, plan, execution)
        if self.llm_service and self.llm_service.status().get("enabled"):
            answer = self._format_with_llm(question, plan, execution, mode, deterministic)
            if answer:
                return answer
        return deterministic

    def _format_deterministic(self, question: str, plan: dict[str, Any], execution: dict[str, Any]) -> str:
        table = execution["table"]
        header = f"## Resposta\n\nConsulta executada na base **{table['filename']} / {table['sheet_name']}**."
        filters_md = self._filters_to_markdown(execution.get("filters", []))
        if execution["type"] == "describe_base":
            return (
                f"{header}\n\n"
                f"A base possui **{execution['rows_considered']} registros** e **{len(execution.get('columns', []))} colunas**. "
                f"Ela contém informações estruturadas e textuais que podem ser consultadas por contagem, listagem, agrupamento ou perguntas documentais.\n\n"
                f"### Principais distribuições\n{self._distributions_to_markdown(execution.get('distributions', {}))}\n\n"
                f"### Amostra\n{self._records_to_markdown(execution.get('sample', []))}"
            )
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
            f"**Registros encontrados:** {execution['rows_filtered']}\n"
            f"**Registros exibidos:** {execution.get('returned_rows', len(execution.get('results', [])))}\n\n"
            f"### Filtros aplicados\n{filters_md}\n\n"
            f"### Registros\n{self._records_to_markdown(execution.get('results', []))}"
        )

    def _format_with_llm(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str, deterministic_answer: str) -> str | None:
        system_prompt = (
            "Você é um analista sênior do GABBI. Responda somente com base no resultado estruturado recebido. "
            "Não invente números, filtros ou linhas. Não diga que não encontrou se o resultado estruturado contém contagem intermediária ou registros. "
            "Não adicione filtros que não estejam em execution.filters. Se houver 0, explique exatamente quais filtros zeraram. "
            "Use markdown claro e objetivo."
        )
        if mode == "executive":
            system_prompt += " Use linguagem executiva e simples."
        elif mode == "technical":
            system_prompt += " Use linguagem técnica e inclua detalhes da execução."
        payload = {
            "pergunta": question,
            "plano_executado": plan,
            "resultado_executado": execution,
            "resposta_deterministica_base": deterministic_answer,
        }
        return self.llm_service.generate_chat(system_prompt, [], f"Formate a resposta a partir deste payload JSON:\n{payload}", temperature=0)

    def _filters_to_markdown(self, filters: list[dict[str, Any]]) -> str:
        if not filters:
            return "- Nenhum filtro específico foi aplicado."
        lines = []
        for f in filters:
            extra = ""
            if "rows_before" in f and "rows_after" in f:
                extra = f" — {f['rows_before']} → {f['rows_after']} linhas"
            lines.append(f"- **{f['column']}** {f['operator']} `{f['value']}`{extra}")
        return "\n".join(lines)

    def _records_to_markdown(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "Nenhum registro encontrado."
        # Priorizar colunas úteis e limitar largura.
        all_cols = list(records[0].keys())
        preferred = ["numero", "codigo_principal", "codigo_tipo", "mes", "tipo", "categoria", "estado", "status", "prioridade", "grupo_atribuicao", "ic_impactado", "topic_name", "article_ref_id"]
        columns = [c for c in preferred if c in all_cols] + [c for c in all_cols if c not in preferred]
        columns = columns[:10]
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in records[:50]:
            vals = [str(row.get(col, "")).replace("\n", " ").replace("|", "/")[:140] for col in columns]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _distributions_to_markdown(self, distributions: dict[str, list[dict[str, Any]]]) -> str:
        if not distributions:
            return "Nenhuma distribuição principal foi calculada."
        parts = []
        for col, rows in distributions.items():
            parts.append(f"#### {col}")
            parts.append(self._records_to_markdown(rows))
        return "\n\n".join(parts)

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
