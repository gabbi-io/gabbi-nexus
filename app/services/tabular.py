from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from app.services.query_intelligence import QueryIntelligenceService

try:
    from sqlalchemy import create_engine, inspect, text
    HAS_SQLALCHEMY = True
except Exception:
    HAS_SQLALCHEMY = False


@dataclass
class TableRef:
    source_path: str
    filename: str
    sheet_name: str
    row_count: int
    columns: list[str]
    source_type: str = "file"  # file | postgres
    schema_name: str | None = None
    table_name: str | None = None


class TabularQueryService:
    """
    Consulta tabular híbrida do Nexus.

    Ajustes aplicados:
    - não prioriza gabbi_knowledge_table_active_*.csv local;
    - ignora CSV cache local do Gabbi como fonte de verdade;
    - mantém planilhas/CSVs anexados pelo usuário como fonte válida do case;
    - adiciona PostgreSQL como fonte viva da base treinada;
    - escolhe a fonte conforme a pergunta: arquivo/anexo vs base/treinamento.
    """

    FILE_FOCUS_TERMS = {
        "arquivo", "arquivos", "anexo", "anexado", "anexados", "documento", "documentos",
        "planilha", "excel", "xlsx", "xls", "csv", "pdf", "upload", "uploads", "enviado", "enviados",
        "neste", "nessa", "nesta", "deste", "dessa", "desta", "material"
    }

    KB_FOCUS_TERMS = {
        "base", "conhecimento", "treinamento", "treinamentos", "treinado", "treinada",
        "article", "artigo", "artigos", "topic", "topico", "tópico", "historico", "histórico",
        "memoria", "memória", "postgres", "banco", "gabbi"
    }

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._catalog_cache: dict[str, list[TableRef]] = {}
        self._query_context: dict[str, dict[str, Any]] = {}
        self.qi = QueryIntelligenceService()

        self.disable_local_knowledge_table = self._env_bool("GABBI_DISABLE_LOCAL_KNOWLEDGE_TABLE", True)
        # Agora este flag significa: usar DB quando houver pergunta de base/treinamento,
        # não significa ignorar anexos do case.
        self.force_db_tabular = self._env_bool("GABBI_FORCE_DB_TABULAR", True)
        self.database_url = (
            os.getenv("GABBI_DATABASE_URL", "").strip()
            or os.getenv("GABBI_POSTGRES_URL", "").strip()
            or os.getenv("DATABASE_URL", "").strip()
        )
        self.db_schema = os.getenv("GABBI_TABULAR_SCHEMA", "public").strip() or "public"
        self.db_table = os.getenv("GABBI_TABULAR_TABLE", "gabbi_knowledge_table_active").strip()
        self.db_max_rows = int(os.getenv("GABBI_TABULAR_MAX_ROWS", "200000"))
        self._db_engine = None

        if HAS_SQLALCHEMY and self.database_url:
            try:
                self._db_engine = create_engine(self.database_url, future=True, pool_pre_ping=True)
            except Exception:
                self._db_engine = None

    @staticmethod
    def _env_bool(name: str, default: bool = False) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "sim", "on"}

    def clear_context(self, case_id: str) -> None:
        self._query_context.pop(case_id, None)

    def build_catalog(self, case_id: str, documents: list[dict[str, Any]]) -> dict[str, Any]:
        tables = self._load_tables(documents)
        self._catalog_cache[case_id] = tables
        self.clear_context(case_id)
        return {
            "tables": [
                {
                    "filename": t.filename,
                    "sheet_name": t.sheet_name,
                    "row_count": t.row_count,
                    "columns": t.columns,
                    "source_type": t.source_type,
                    "schema_name": t.schema_name,
                    "table_name": t.table_name,
                }
                for t in tables
            ],
            "tables_count": len(tables),
            "csv_cache_disabled": self.disable_local_knowledge_table,
            "db_tabular_enabled": bool(self._db_engine is not None),
        }

    def answer_question(
        self,
        case_id: str,
        question: str,
        documents: list[dict[str, Any]],
        mode: str = "executive",
        source_preference: str | None = None,
    ) -> dict[str, Any] | None:
        tables = self._catalog_cache.get(case_id) or self._load_tables(documents)
        self._catalog_cache[case_id] = tables

        if not tables:
            return None

        target = self._pick_best_table(question, tables, source_preference=source_preference)
        if not target:
            return None

        previous = self._query_context.get(case_id)
        plan = self.qi.build_plan(question, target.columns, previous_plan=previous)
        if plan.reset_context:
            self.clear_context(case_id)
            plan = self.qi.build_plan(question, target.columns, previous_plan=None)

        if not plan.use_tabular:
            return None

        if self._is_forbidden_local_cache(target):
            return {
                "route": "tabular",
                "query_type": plan.intent,
                "fallback_to_rag": True,
                "answer_text": "",
                "summary": "",
                "technical": {
                    "plan": self.qi.plan_to_dict(plan),
                    "reason": "local_csv_cache_disabled",
                    "blocked_file": target.filename,
                },
                "evidences": [],
                "evidence_files": [],
            }

        execution = self._execute_plan(plan, target)
        plan_dict = self.qi.plan_to_dict(plan)

        if execution.get("success") and execution.get("rows_filtered", 0) == 0:
            inherited_count = sum(1 for f in plan.filters if f.source == "inherited")
            explicit_count = sum(1 for f in plan.filters if f.source == "explicit")
            if inherited_count > 0 or explicit_count == 0:
                return {
                    "route": "tabular",
                    "query_type": plan.intent,
                    "fallback_to_rag": True,
                    "answer_text": "",
                    "summary": "",
                    "technical": {"plan": plan_dict, "execution": execution},
                    "evidences": execution.get("evidences", []),
                    "evidence_files": execution.get("evidence_files", []),
                }

        if not execution.get("success"):
            return {
                "route": "tabular",
                "query_type": plan.intent,
                "fallback_to_rag": True,
                "answer_text": execution.get("message", ""),
                "summary": execution.get("message", ""),
                "technical": {"plan": plan_dict, "execution": execution},
                "evidences": [],
                "evidence_files": [],
            }

        self._query_context[case_id] = {
            **plan_dict,
            "last_execution_summary": {
                "rows_considered": execution.get("rows_considered"),
                "rows_filtered": execution.get("rows_filtered"),
                "type": execution.get("type"),
                "source_type": target.source_type,
            },
        }

        answer = self._format_answer(question, plan_dict, execution, mode)
        return {
            "route": "tabular_db" if target.source_type == "postgres" else "tabular_file",
            "query_type": plan.intent,
            "answer_text": answer,
            "summary": answer,
            "technical": {"plan": plan_dict, "execution": execution},
            "evidences": execution.get("evidences", []),
            "evidence_files": execution.get("evidence_files", []),
        }

    def _pick_best_table(
        self,
        question: str,
        tables: list[TableRef],
        source_preference: str | None = None,
    ) -> TableRef | None:
        if not tables:
            return None

        allowed = [t for t in tables if not self._is_forbidden_local_cache(t)]
        if not allowed:
            return None

        db_tables = [t for t in allowed if t.source_type == "postgres"]
        file_tables = [t for t in allowed if t.source_type == "file"]

        question_focus = source_preference or self._detect_focus(question)

        # Se usuário mencionou arquivo/anexo/planilha, privilegia anexo do case.
        if question_focus in {"case_upload_first", "hybrid_case_first"} and file_tables:
            ranked_files = self._rank_tables(question, file_tables, prefer="file")
            # Se houver match minimamente plausível em arquivo, usa arquivo.
            if ranked_files:
                return ranked_files[0]

        # Se usuário mencionou base/treinamento ou não há arquivo tabular, privilegia DB.
        if db_tables:
            ranked_db = self._rank_tables(question, db_tables, prefer="postgres")
            if question_focus == "knowledge_base_first" or not file_tables:
                return ranked_db[0]

        # Fallback para arquivo do case.
        if file_tables:
            return self._rank_tables(question, file_tables, prefer="file")[0]

        return db_tables[0] if db_tables else None

    def _detect_focus(self, question: str) -> str:
        q = self._norm(question)
        tokens = set(q.split())
        file_focus = bool(tokens & self.FILE_FOCUS_TERMS)
        kb_focus = bool(tokens & self.KB_FOCUS_TERMS)
        if file_focus:
            return "case_upload_first"
        if kb_focus:
            return "knowledge_base_first"
        return "hybrid_case_first"

    def _rank_tables(self, question: str, tables: list[TableRef], prefer: str | None = None) -> list[TableRef]:
        question_norm = self._norm(question)
        tokens = [token for token in question_norm.split() if len(token) > 3]

        def score(table: TableRef) -> tuple[float, int]:
            cols = " ".join(self._norm(c) for c in table.columns)
            filename = self._norm(table.filename)
            overlap = sum(1 for token in tokens if token in cols or token in filename)
            base = float(overlap * 5)

            if prefer == table.source_type:
                base += 10
            if table.source_type == "file" and any(token in filename for token in ["upload", "anexo", "arquivo"]):
                base += 4
            if table.source_type == "postgres":
                base += 2

            return (base, table.row_count)

        return sorted(tables, key=score, reverse=True)

    def _execute_plan(self, plan, table: TableRef) -> dict[str, Any]:
        try:
            df = self._load_dataframe(table)
        except Exception as exc:
            return {"success": False, "message": f"Erro ao carregar fonte tabular: {str(exc)}"}

        if df.empty:
            return {"success": False, "message": "A fonte tabular está vazia."}

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
            needle = str(value)

            if op == "eq":
                filtered = filtered[series.str.lower() == needle.lower()]
            elif op == "startswith":
                filtered = filtered[series.str.lower().str.startswith(needle.lower(), na=False)]
            else:
                filtered = filtered[series.str.contains(re.escape(needle), case=False, na=False)]

            after = int(len(filtered))
            applied_filters.append(
                {
                    "column": column,
                    "operator": op,
                    "value": value,
                    "source": filt.source,
                    "before": before,
                    "after": after,
                }
            )

        source_label = self._source_label(table)
        evidences = [
            {
                "filename": source_label,
                "sheet_name": table.sheet_name,
                "score": 1.0,
                "excerpt": f"Fonte {source_label} com {table.row_count} linhas e {len(table.columns)} colunas.",
                "source_type": table.source_type,
            }
        ]

        if plan.intent == "count":
            return {
                "success": True,
                "type": "count",
                "count": int(len(filtered)),
                "rows_considered": int(len(df)),
                "rows_filtered": int(len(filtered)),
                "table": self._table_payload(table),
                "filters": applied_filters,
                "preview": self._safe_records(filtered.head(8)),
                "evidences": evidences,
                "evidence_files": [source_label],
            }

        if plan.intent == "group":
            group_by = plan.group_by
            if not group_by or group_by not in filtered.columns:
                return {"success": False, "message": "Não consegui identificar a coluna para agrupar a consulta."}
            grouped = filtered[group_by].astype(str).fillna("").value_counts(dropna=False).reset_index()
            grouped.columns = [group_by, "total"]
            return {
                "success": True,
                "type": "group",
                "rows_considered": int(len(df)),
                "rows_filtered": int(len(filtered)),
                "table": self._table_payload(table),
                "filters": applied_filters,
                "group_by": group_by,
                "results": self._safe_records(grouped.head(50)),
                "evidences": evidences,
                "evidence_files": [source_label],
            }

        limit = int(plan.limit or 20)
        return {
            "success": True,
            "type": "list",
            "rows_considered": int(len(df)),
            "rows_filtered": int(len(filtered)),
            "table": self._table_payload(table),
            "filters": applied_filters,
            "results": self._safe_records(filtered.head(limit)),
            "columns": list(filtered.columns),
            "evidences": evidences,
            "evidence_files": [source_label],
        }

    def _format_answer(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str:
        if self.llm_service and self.llm_service.status().get("enabled"):
            answer = self._format_with_llm(question, plan, execution, mode)
            if answer:
                return answer

        table = execution["table"]
        filters_md = self._filters_to_markdown(execution.get("filters", []))
        source_name = table.get("source") or f"{table.get('filename')} / {table.get('sheet_name')}"
        source_type = table.get("source_type")
        source_label = "arquivo anexado" if source_type == "file" else "base viva PostgreSQL"
        header = f"## Resposta\n\nConsulta executada na fonte **{source_name}** ({source_label})."

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
            f"**Registros encontrados:** {execution['rows_filtered']}\n\n"
            f"### Filtros aplicados\n{filters_md}\n\n"
            f"### Registros\n{self._records_to_markdown(execution.get('results', []))}"
        )

    def _format_with_llm(self, question: str, plan: dict[str, Any], execution: dict[str, Any], mode: str) -> str | None:
        system_prompt = (
            "Você é um analista sênior do GABBI. Receba o resultado de uma consulta tabular já executada "
            "e redija a resposta em markdown. Nunca invente números. Use exatamente os dados de execution. "
            "Explique cobertura, filtros, resultado e se a fonte foi arquivo anexado ou base viva PostgreSQL. "
            "Não mencione CSV/cache local como base consultada."
        )
        if mode == "executive":
            system_prompt += " Use linguagem executiva e direta."
        user_prompt = f"Pergunta: {question}\n\nPlano: {plan}\n\nResultado executado: {execution}"
        return self.llm_service.generate_chat(system_prompt, [], user_prompt, temperature=0)

    def _filters_to_markdown(self, filters: list[dict[str, Any]]) -> str:
        if not filters:
            return "- Nenhum filtro específico foi aplicado."
        return "\n".join(
            [
                f"{i}. `{f.get('column')}` {f.get('operator')} `{f.get('value')}` — {f.get('before')} → {f.get('after')} linhas"
                for i, f in enumerate(filters, start=1)
            ]
        )

    def _records_to_markdown(self, records: list[dict[str, Any]]) -> str:
        if not records:
            return "Nenhum registro encontrado."
        preferred = [
            "numero", "codigo_principal", "codigo_tipo", "mes", "tipo", "estado", "status",
            "prioridade", "grupo_atribuicao", "ic_impactado",
        ]
        cols = [c for c in preferred if c in records[0]] or list(records[0].keys())[:8]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for row in records[:12]:
            vals = [str(row.get(col, "")).replace("\n", " ")[:120] for col in cols]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def _load_tables(self, documents: list[dict[str, Any]]) -> list[TableRef]:
        tables: list[TableRef] = []

        # 1. Carrega anexos tabulares do case atual.
        for doc in documents:
            path = Path(doc.get("path", ""))
            if not path.exists():
                continue
            suffix = path.suffix.lower()
            try:
                if suffix == ".csv":
                    if self.disable_local_knowledge_table and self._looks_like_knowledge_cache(path.name):
                        continue
                    df = pd.read_csv(path, dtype=str, keep_default_na=False)
                    tables.append(
                        TableRef(
                            str(path), path.name, "csv", int(df.shape[0]),
                            [str(c).strip() for c in df.columns.tolist()], source_type="file",
                        )
                    )
                elif suffix in {".xlsx", ".xlsm", ".xls"}:
                    xl = pd.ExcelFile(path)
                    for sheet in xl.sheet_names:
                        df = xl.parse(sheet, dtype=str).fillna("")
                        tables.append(
                            TableRef(
                                str(path), path.name, sheet, int(df.shape[0]),
                                [str(c).strip() for c in df.columns.tolist()], source_type="file",
                            )
                        )
            except Exception:
                continue

        # 2. Carrega referência da base viva PostgreSQL, se disponível.
        db_ref = self._load_postgres_table_ref()
        if db_ref:
            tables.append(db_ref)

        return tables

    def _load_postgres_table_ref(self) -> TableRef | None:
        if not self._db_engine or not self.db_table:
            return None
        try:
            inspector = inspect(self._db_engine)
            table_names = inspector.get_table_names(schema=self.db_schema)
            if self.db_table not in table_names:
                return None
            columns = [c["name"] for c in inspector.get_columns(self.db_table, schema=self.db_schema)]
            quoted_schema = self._quote_ident(self.db_schema)
            quoted_table = self._quote_ident(self.db_table)
            sql = text(f"SELECT COUNT(*) FROM {quoted_schema}.{quoted_table}")
            with self._db_engine.connect() as conn:
                row_count = int(conn.execute(sql).scalar() or 0)
            return TableRef(
                source_path=f"postgres://{self.db_schema}.{self.db_table}",
                filename=f"{self.db_schema}.{self.db_table}",
                sheet_name="postgres",
                row_count=row_count,
                columns=[str(c).strip() for c in columns],
                source_type="postgres",
                schema_name=self.db_schema,
                table_name=self.db_table,
            )
        except Exception:
            return None

    def _load_dataframe(self, table: TableRef) -> pd.DataFrame:
        if table.source_type == "postgres":
            return self._load_postgres_dataframe(table)

        path = Path(table.source_path)
        if path.suffix.lower() == ".csv":
            if self._is_forbidden_local_cache(table):
                raise RuntimeError("CSV/cache local do Gabbi desabilitado como fonte de consulta.")
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
        else:
            df = pd.read_excel(path, sheet_name=table.sheet_name, dtype=str).fillna("")
        df.columns = [str(c).strip() for c in df.columns]
        return df.fillna("")

    def _load_postgres_dataframe(self, table: TableRef) -> pd.DataFrame:
        if not self._db_engine or not table.table_name:
            raise RuntimeError("PostgreSQL tabular não configurado.")
        quoted_schema = self._quote_ident(table.schema_name or self.db_schema)
        quoted_table = self._quote_ident(table.table_name)
        sql = f"SELECT * FROM {quoted_schema}.{quoted_table} LIMIT {int(self.db_max_rows)}"
        df = pd.read_sql_query(sql, self._db_engine, dtype=str).fillna("")
        df.columns = [str(c).strip() for c in df.columns]
        return df.fillna("")

    def _safe_records(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        return df.fillna("").astype(str).to_dict(orient="records")

    def _table_payload(self, table: TableRef) -> dict[str, Any]:
        return {
            "filename": table.filename,
            "sheet_name": table.sheet_name,
            "source_type": table.source_type,
            "source": self._source_label(table),
            "schema_name": table.schema_name,
            "table_name": table.table_name,
        }

    def _source_label(self, table: TableRef) -> str:
        if table.source_type == "postgres":
            return f"PostgreSQL:{table.schema_name or self.db_schema}.{table.table_name or self.db_table}"
        return f"Arquivo:{table.filename}/{table.sheet_name}"

    def _is_forbidden_local_cache(self, table: TableRef) -> bool:
        return bool(
            self.disable_local_knowledge_table
            and table.source_type == "file"
            and self._looks_like_knowledge_cache(table.filename)
        )

    @staticmethod
    def _looks_like_knowledge_cache(filename: str) -> bool:
        name = filename.lower()
        return name.startswith("gabbi_knowledge_table_active") or "gabbi_knowledge_table_active" in name

    @staticmethod
    def _quote_ident(identifier: str) -> str:
        safe = identifier.replace('"', '""')
        return f'"{safe}"'

    @staticmethod
    def _norm(value: Any) -> str:
        text_value = "" if value is None else str(value)
        text_value = text_value.lower().strip()
        text_value = re.sub(r"[^a-z0-9_:/\-\s\.áàâãéêíóôõúç]+", " ", text_value)
        return " ".join(text_value.split())
