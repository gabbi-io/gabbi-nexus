from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

try:
    import duckdb
    HAS_DUCKDB = True
except Exception:
    duckdb = None
    HAS_DUCKDB = False


def _safe(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _safe(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
    return " ".join(text.split())


def _upper(value: Any) -> str:
    return _safe(value).upper().strip()


def _json_loads_maybe(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _extract_field(text: str, label: str) -> str:
    """
    Parser robusto para campos no formato "Label: valor".

    Evita vazamento de um campo para o próximo, por exemplo:
    Estado: Encerrado(a)
    Data de início planejada: 2025-...

    Também preserva campos multilinha até encontrar outro label.
    """
    if not text:
        return ""

    label_norm = _norm(label)
    lines = str(text).splitlines()
    collecting = False
    collected: list[str] = []

    label_pattern = re.compile(r"^\s*([^:]{1,80})\s*:\s*(.*)$")

    for line in lines:
        current = line.strip()
        match = label_pattern.match(current)

        if not collecting:
            if match and _norm(match.group(1)) == label_norm:
                collecting = True
                value = match.group(2).strip()
                if value:
                    collected.append(value)
            continue

        if match:
            break

        if current:
            collected.append(current)

    return "\n".join(collected).strip()


def _extract_numero(text: str) -> str:
    direct = _extract_field(text, "Número")
    if direct:
        return _upper(direct)
    match = re.search(r"\b(CHG\d{5,}|INC\d{5,})\b", text or "", re.IGNORECASE)
    return _upper(match.group(1)) if match else ""


def _extract_mes(text: str) -> str:
    direct = _extract_field(text, "Mês")
    target = direct or text or ""
    match = re.search(r"(20\d{2})[-/](\d{1,2})", target)
    if match:
        return f"{match.group(1)}-{match.group(2).zfill(2)}"
    return ""


def _extract_day(value: str) -> str:
    match = re.search(r"(20\d{2})[-/](\d{1,2})[-/](\d{1,2})", value or "")
    return match.group(3).zfill(2) if match else ""


def _sql_ident(col: str) -> str:
    return '"' + col.replace('"', '""') + '"'


def _looks_like_assignment_group(value: str) -> bool:
    value = _upper(value)
    return bool(re.fullmatch(r"[A-Z0-9]+(?:_[A-Z0-9]+){1,}", value))


def _duration_to_seconds(value: Any) -> int:
    """Converte durações HH:MM:SS, H:MM:SS ou MM:SS para segundos."""
    text = _safe(value)
    if not text:
        return 0
    match = re.search(r"(\d{1,4}):(\d{2}):(\d{2})", text)
    if match:
        return int(match.group(1)) * 3600 + int(match.group(2)) * 60 + int(match.group(3))
    match = re.search(r"(\d{1,4}):(\d{2})", text)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    return 0


def _seconds_to_hhmmss(seconds: int | float | None) -> str:
    seconds = int(seconds or 0)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _safe(value)
        if text:
            return text
    return ""


_FIELD_STOP_LABELS = [
    "Mês", "Categoria", "Tipo", "Estado", "Status", "Data de início planejada",
    "Data de termino planejada", "Data de término planejada", "IC Impactado",
    "Grupo de atribuição", "Tipo de Indisponibilidade", "Solicitação de",
    "Aberta por", "Atribuído a", "Descrição resumida", "Descrição", "Tipo de teste",
    "Plano de teste", "Código de fechamento", "Anotações de encerramento", "Aberto",
    "Resolvido", "Encerrado", "Prioridade", "Causa Origem", "Causado pela mudança",
    "Canal impactado", "Canal Impactado", "Tempo total de impacto", "Tempo de impacto",
    "u_rpt_tempo_total_de_impacto",
]


def _clean_extracted_value(value: Any) -> str:
    """Remove rótulos que vazaram dentro do valor extraído."""
    text = _safe(value)
    if not text:
        return ""
    for label in _FIELD_STOP_LABELS:
        m = re.search(rf"\s+{re.escape(label)}\s*:\s*", text, flags=re.IGNORECASE)
        if m:
            text = text[:m.start()].strip()
    return re.sub(r"\s+", " ", text).strip()


def _truthy_text(value: Any) -> bool:
    return bool(_safe(value)) and _norm(value) not in {"-", "none", "null", "na", "n/a"}


class KnowledgeStructuredStore:
    TABLE = "knowledge_articles_structured"

    ANALYTIC_TERMS = {
        "quantos", "quantas", "quantidade", "qtd", "qtde", "total",
        "quais", "qual", "liste", "listar", "lista", "codigos", "códigos",
        "grupo de atribuicao", "grupo de atribuição", "ic impactado",
        "impactou", "referencia", "referência", "causado",
        "change", "changes", "chg", "mudanca", "mudança",
        "incidente", "incidentes", "inc",
        "app", "ecomm", "aura", "whatsapp", "dia", "mes", "mês",
        "aberta", "abertas", "aberto", "abertos",
        "detalhe", "detalhar", "detalhes", "explique", "explicar",
        "resuma", "resumo", "descreva", "descricao", "descrição",
        "plano", "rollback", "teste", "janela", "indisponibilidade",
        "percentual", "porcentagem", "taxa", "sucesso", "bem sucedido",
        "bem-sucedido", "bem sucedidas", "bem-sucedidas",
        "frequente", "frequentes", "ranking", "mais comum", "mais comuns",
        "estado", "estados", "status", "cancelada", "canceladas", "cancelado",
        "cancelados", "encerrada", "encerradas", "encerrado", "encerrados",
        "emergencia", "emergência", "normal",
        "temos", "tem", "existe", "existem", "alguma", "algum",
        "temos alguma", "tem alguma", "existe alguma", "existe algum",
        "kpi", "kpis", "indicador", "indicadores", "executivo", "executiva",
        "faq", "treinamento", "operacao", "operação", "critico", "crítico",
        "criticos", "críticos", "p1", "p2", "p3", "prioridade",
        "impacto", "maior impacto", "tempo de impacto", "tempo total",
        "parada", "sistemica", "sistêmica", "tela de manutencao", "tela de manutenção",
        "mttr", "tempo medio", "tempo médio", "solucao", "solução",
        "causa", "causas", "causa origem", "origem", "funcionalidade",
        "funcionalidades", "comparativo", "compare", "top", "ranking",
        "checkout", "fatura", "faturas", "troca de plano", "recarga", "esim",
        "login", "ordem de servico", "ordem de serviço", "para voce", "pra voce",
        "para você", "pra você", "meu vivo", "app vivo", "aplicativo vivo",
    }

    BASE_COLUMNS: dict[str, str] = {
        "case_id": "TEXT",
        "knowledge_version": "TEXT",
        "agent_key": "TEXT",
        "agent_name": "TEXT",
        "project_key": "TEXT",
        "project_id": "TEXT",
        "project_name": "TEXT",
        "topic_id": "TEXT",
        "topic_ref_id": "TEXT",
        "topic_name": "TEXT",
        "topic_description": "TEXT",
        "article_id": "TEXT",
        "article_ref_id": "TEXT",
        "source_id": "TEXT",
        "codigo_tipo": "TEXT",
        "codigo_principal": "TEXT",
        "numero": "TEXT",
        "transacao_z": "TEXT",
        "mes": "TEXT",
        "tipo": "TEXT",
        "estado": "TEXT",
        "status": "TEXT",
        "grupo_atribuicao": "TEXT",
        "ic_impactado": "TEXT",
        "canal": "TEXT",
        "categoria": "TEXT",
        "prioridade": "TEXT",
        "data_inicio_planejada": "TEXT",
        "data_termino_planejada": "TEXT",
        "article_updated_on": "TEXT",
        "article_text": "TEXT",
        "raw_json": "TEXT",

        # Colunas adicionais preservam dados que antes só ficavam no article_text/raw_json.
        # A criação é retrocompatível: _ensure_schema adiciona se a tabela já existir.
        "dia": "TEXT",
        "is_app": "BOOLEAN",
        "is_ecomm": "BOOLEAN",
        "descricao": "TEXT",
        "descricao_resumida": "TEXT",
        "causado_pela_mudanca": "TEXT",
        "aberto": "TEXT",
        "tempo_impacto": "TEXT",
        "tempo_impacto_segundos": "BIGINT",
        "causa_origem": "TEXT",
        "resolvido": "TEXT",
        "encerrado": "TEXT",
        "funcionalidade": "TEXT",
        "is_parada_sistemica": "BOOLEAN",
        "is_change_related": "BOOLEAN",

        "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
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

    def _connect(self):
        if not self.enabled:
            raise RuntimeError("DuckDB não instalado")
        return duckdb.connect(str(self.db_path))

    def _ensure_schema(self) -> None:
        cols_sql = ",\n".join(
            f"{_sql_ident(col)} {definition}"
            for col, definition in self.BASE_COLUMNS.items()
        )
        with self._connect() as con:
            con.execute(f"CREATE TABLE IF NOT EXISTS {self.TABLE} ({cols_sql})")

            # Retrocompatibilidade para tabelas já criadas antes deste ajuste.
            existing = set(self._table_cols(con))
            for col, definition in self.BASE_COLUMNS.items():
                if col not in existing:
                    con.execute(f"ALTER TABLE {self.TABLE} ADD COLUMN {_sql_ident(col)} {definition}")

    def _table_cols(self, con) -> list[str]:
        return [row[1] for row in con.execute(f"PRAGMA table_info('{self.TABLE}')").fetchall()]

    def status(self) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False, "reason": "duckdb_not_installed"}
        try:
            with self._connect() as con:
                total = con.execute(f"SELECT COUNT(*) FROM {self.TABLE}").fetchone()[0]
                cases = con.execute(f"SELECT COUNT(DISTINCT case_id) FROM {self.TABLE}").fetchone()[0]
                cols = self._table_cols(con)
            return {"enabled": True, "db_path": str(self.db_path), "rows": int(total), "cases": int(cases), "columns": cols}
        except Exception as exc:
            return {"enabled": False, "error": str(exc), "db_path": str(self.db_path)}

    def replace_case_rows(self, case_id: str, rows: list[dict[str, Any]], knowledge_version: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("DuckDB não está instalado. Instale com: pip install duckdb")

        normalized = [self._normalize_row(case_id, row, knowledge_version) for row in (rows or [])]

        with self._connect() as con:
            table_cols = self._table_cols(con)
            table_col_set = set(table_cols)

            if "case_id" in table_col_set:
                con.execute(f"DELETE FROM {self.TABLE} WHERE case_id = ?", [case_id])
            else:
                con.execute(f"DELETE FROM {self.TABLE}")

            if normalized:
                insert_cols = [col for col in normalized[0].keys() if col in table_col_set and col != "updated_at"]
                if not insert_cols:
                    raise RuntimeError(f"Nenhuma coluna compatível encontrada. Tabela={table_cols}; Normalizado={list(normalized[0].keys())}")

                cols_sql = ", ".join(_sql_ident(c) for c in insert_cols)
                placeholders = ", ".join(["?"] * len(insert_cols))
                sql = f"INSERT INTO {self.TABLE} ({cols_sql}) VALUES ({placeholders})"
                values = [[row.get(col) for col in insert_cols] for row in normalized]
                con.executemany(sql, values)

        self.memory.pop(case_id, None)
        return {"success": True, "case_id": case_id, "rows_received": len(rows or []), "rows_saved": len(normalized), "knowledge_version": knowledge_version}

    def _normalize_row(self, case_id: str, row: dict[str, Any], knowledge_version: str | None) -> dict[str, Any]:
        raw_json = json.dumps(row, ensure_ascii=False, default=str)
        document_obj = _json_loads_maybe(row.get("document"))

        article_text = _safe(row.get("article") or row.get("article_text") or row.get("text") or "")
        if not article_text:
            article_text = raw_json

        numero = (
            _upper(row.get("numero"))
            or _upper(row.get("Número"))
            or _upper(row.get("codigo_principal"))
            or _upper(document_obj.get("Número"))
            or _upper(document_obj.get("numero"))
            or _extract_numero(article_text)
        )

        if numero.startswith("CHG"):
            codigo_tipo = "CHG"
        elif numero.startswith("INC"):
            codigo_tipo = "INC"
        else:
            codigo_tipo = _upper(row.get("codigo_tipo")) or _upper(row.get("categoria")) or "OTHER"

        categoria = _upper(row.get("categoria")) or codigo_tipo

        mes = (
            _safe(row.get("mes"))
            or _safe(row.get("Mês"))
            or _safe(document_obj.get("Mês"))
            or _safe(document_obj.get("mes"))
            or _extract_mes(article_text)
        )
        mes = self._normalize_month(mes)

        data_inicio = _safe(row.get("data_inicio_planejada")) or _safe(document_obj.get("Data de início planejada")) or _extract_field(article_text, "Data de início planejada")
        data_termino = _safe(row.get("data_termino_planejada")) or _safe(document_obj.get("Data de término planejada")) or _extract_field(article_text, "Data de término planejada")
        aberto = _safe(row.get("aberto")) or _safe(document_obj.get("Aberto")) or _extract_field(article_text, "Aberto")

        if not mes:
            mes = self._normalize_month(data_inicio if codigo_tipo == "CHG" else aberto)

        dia = _extract_day(data_inicio if codigo_tipo == "CHG" else aberto)

        ic = _safe(row.get("ic_impactado")) or _safe(document_obj.get("IC Impactado")) or _extract_field(article_text, "IC Impactado")
        grupo = _safe(row.get("grupo_atribuicao")) or _safe(row.get("grupo_de_atribuicao")) or _safe(document_obj.get("Grupo de atribuição")) or _extract_field(article_text, "Grupo de atribuição")
        canal = (
            _safe(row.get("canal")) or _safe(row.get("canal_impactado"))
            or _safe(document_obj.get("Canal impactado")) or _safe(document_obj.get("Canal Impactado"))
            or _extract_field(article_text, "Canal impactado") or _extract_field(article_text, "Canal Impactado")
        )
        descricao_resumida = _safe(row.get("descricao_resumida")) or _safe(document_obj.get("Descrição resumida")) or _extract_field(article_text, "Descrição resumida")
        descricao = _safe(row.get("descricao")) or _safe(document_obj.get("Descrição")) or _extract_field(article_text, "Descrição")
        causado = _safe(row.get("causado_pela_mudanca")) or _safe(document_obj.get("Causado pela mudança")) or _extract_field(article_text, "Causado pela mudança")
        estado = _safe(row.get("estado")) or _safe(document_obj.get("Estado")) or _extract_field(article_text, "Estado")
        tipo = _safe(row.get("tipo")) or _safe(document_obj.get("Tipo")) or _extract_field(article_text, "Tipo")
        prioridade = _safe(row.get("prioridade")) or _safe(document_obj.get("Prioridade")) or _extract_field(article_text, "Prioridade")
        tempo_impacto = _first_non_empty(
            row.get("tempo_impacto"),
            row.get("u_rpt_tempo_total_de_impacto"),
            row.get("Tempo total de impacto"),
            row.get("Tempo de impacto"),
            row.get("tempo_total_impacto"),
            document_obj.get("u_rpt_tempo_total_de_impacto"),
            document_obj.get("Tempo total de impacto"),
            document_obj.get("Tempo de impacto"),
            _extract_field(article_text, "u_rpt_tempo_total_de_impacto"),
            _extract_field(article_text, "Tempo total de impacto"),
            _extract_field(article_text, "Tempo de impacto"),
        )
        causa_origem = _first_non_empty(
            row.get("causa_origem"),
            row.get("Causa Origem"),
            row.get("causa origem"),
            document_obj.get("Causa Origem"),
            document_obj.get("causa_origem"),
            _extract_field(article_text, "Causa Origem"),
            _extract_field(article_text, "Causa origem"),
        )
        resolvido = _first_non_empty(row.get("resolvido"), document_obj.get("Resolvido"), _extract_field(article_text, "Resolvido"))
        encerrado = _first_non_empty(row.get("encerrado"), document_obj.get("Encerrado"), _extract_field(article_text, "Encerrado"))

        # Limpeza defensiva contra vazamento de rótulos em campos curtos.
        estado = _clean_extracted_value(estado)
        status = estado
        tipo = _clean_extracted_value(tipo)
        prioridade = _clean_extracted_value(prioridade)
        grupo = _clean_extracted_value(grupo)
        ic = _clean_extracted_value(ic)
        canal = _clean_extracted_value(canal)
        causado = _clean_extracted_value(causado)
        causa_origem = _clean_extracted_value(causa_origem)
        descricao_resumida = _clean_extracted_value(descricao_resumida)

        blob = _norm(" ".join([canal, descricao_resumida, descricao, article_text, raw_json]))
        is_app = bool(
            '"is_app": true' in raw_json.lower()
            or '"is_app":true' in raw_json.lower()
            or "canal impactado: app vivo" in blob
            or "canal impactado app vivo" in blob
            or "app vivo" in _norm(canal)
        )
        is_ecomm = bool(
            '"is_ecomm": true' in raw_json.lower()
            or '"is_ecomm":true' in raw_json.lower()
            or "ecomm" in blob
            or "e commerce" in blob
            or "ecommerce" in blob
            or "loja online" in blob
        )

        funcionalidade = _first_non_empty(
            row.get("funcionalidade"),
            row.get("Funcionalidade"),
            document_obj.get("Funcionalidade"),
            ic,
        )
        is_parada_sistemica = bool(
            "indisponibilidade" in blob
            or "tela de manutencao" in blob
            or "tela de manutenção" in blob
        )
        is_change_related = bool(
            _truthy_text(causado)
            or "mudanca" in _norm(causa_origem)
            or "mudança" in _norm(causa_origem)
            or "causado pela mudanca" in blob
            or "causado pela mudança" in blob
        )

        canal_enriched = canal
        if is_app and "APP_MARKER" not in canal_enriched:
            canal_enriched = (canal_enriched + " APP_MARKER").strip()
        if is_ecomm and "ECOMM_MARKER" not in canal_enriched:
            canal_enriched = (canal_enriched + " ECOMM_MARKER").strip()

        article_text_enriched = article_text
        if causado and "CAUSADO_PELA_MUDANCA_MARKER" not in article_text_enriched:
            article_text_enriched += f"\nCAUSADO_PELA_MUDANCA_MARKER: {causado}"
        if dia:
            article_text_enriched += f"\nDIA_NORMALIZADO_MARKER: {dia}"
        if is_app:
            article_text_enriched += "\nAPP_MARKER: true"
        if is_ecomm:
            article_text_enriched += "\nECOMM_MARKER: true"
        if is_parada_sistemica:
            article_text_enriched += "\nPARADA_SISTEMICA_MARKER: true"
        if is_change_related:
            article_text_enriched += "\nCHANGE_RELATED_MARKER: true"

        return {
            "case_id": case_id,
            "knowledge_version": knowledge_version,
            "agent_key": _safe(row.get("agent_key")),
            "agent_name": _safe(row.get("agent_name")),
            "project_key": _safe(row.get("project_key")),
            "project_id": _safe(row.get("project_id")),
            "project_name": _safe(row.get("project_name")),
            "topic_id": _safe(row.get("topic_id") or row.get("topicId")),
            "topic_ref_id": _safe(row.get("topic_ref_id")),
            "topic_name": _safe(row.get("topic_name")),
            "topic_description": _safe(row.get("topic_description")),
            "article_id": _safe(row.get("article_id") or row.get("id")),
            "article_ref_id": _safe(row.get("article_ref_id") or row.get("ref_id") or row.get("refId")),
            "source_id": _safe(row.get("source_id") or row.get("source")),
            "codigo_tipo": codigo_tipo,
            "codigo_principal": numero,
            "numero": numero,
            "transacao_z": _safe(row.get("transacao_z")),
            "mes": mes,
            "tipo": tipo,
            "estado": estado,
            "status": estado,
            "grupo_atribuicao": grupo,
            "ic_impactado": ic,
            "canal": canal_enriched,
            "categoria": categoria,
            "prioridade": prioridade,
            "data_inicio_planejada": data_inicio,
            "data_termino_planejada": data_termino,
            "article_updated_on": _safe(row.get("article_updated_on") or row.get("updated_on") or row.get("updatedOn")),
            "article_text": article_text_enriched,
            "raw_json": raw_json,
            "dia": dia,
            "is_app": is_app,
            "is_ecomm": is_ecomm,
            "descricao": descricao,
            "descricao_resumida": descricao_resumida,
            "causado_pela_mudanca": causado,
            "aberto": aberto,
            "tempo_impacto": tempo_impacto,
            "tempo_impacto_segundos": _duration_to_seconds(tempo_impacto),
            "causa_origem": causa_origem,
            "resolvido": resolvido,
            "encerrado": encerrado,
            "funcionalidade": funcionalidade,
            "is_parada_sistemica": is_parada_sistemica,
            "is_change_related": is_change_related,
        }

    def answer_question(self, case_id: str, question: str, chat_history: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        q = _norm(question)

        # Só perguntas realmente candidatas entram na camada estruturada.
        # Conversas abertas continuam indo para o RAG.
        is_structured_candidate = any(term in q for term in self.ANALYTIC_TERMS)
        if not is_structured_candidate:
            return {"fallback_to_rag": True}

        # Se a pergunta é analítica/operacional, NUNCA deixe cair silenciosamente no RAG,
        # porque o RAG estava respondendo textos genéricos do tipo "Os dados indicam...".
        if not self.enabled:
            return {
                "fallback_to_rag": False,
                "route": "knowledge_structured_disabled",
                "query_type": "structured_unavailable",
                "answer_text": "Não consegui consultar a base estruturada porque o DuckDB não está disponível neste ambiente.",
                "summary": "Não consegui consultar a base estruturada porque o DuckDB não está disponível neste ambiente.",
                "technical": {"case_id": case_id, "reason": "duckdb_not_installed_or_import_failed"},
                "sources": {"deterministic": True, "engine": "duckdb", "table": self.TABLE},
            }

        try:
            result = self._answer_structured(case_id, question)
            if result and result.get("fallback_to_rag"):
                # Proteção anti-alucinação: se era pergunta estruturada e a engine não conseguiu resolver,
                # não delega para o RAG genérico.
                return {
                    "fallback_to_rag": False,
                    "route": "knowledge_structured_no_match",
                    "query_type": "structured_no_match",
                    "answer_text": "Não consegui calcular essa pergunta pela base estruturada disponível. Verifique se o case foi sincronizado/reindexado e se os campos necessários existem.",
                    "summary": "Não consegui calcular essa pergunta pela base estruturada disponível.",
                    "technical": {"case_id": case_id, "question": question, "original_result": result},
                    "sources": {"deterministic": True, "engine": "duckdb", "table": self.TABLE},
                }
            return result
        except Exception as exc:
            # Em pergunta estruturada, erro deve aparecer de forma determinística para diagnóstico,
            # em vez de cair no RAG e parecer resposta válida.
            return {
                "fallback_to_rag": False,
                "route": "knowledge_structured_error_visible",
                "query_type": "structured_error",
                "answer_text": f"Erro ao consultar a base estruturada: {type(exc).__name__}: {exc}",
                "summary": f"Erro ao consultar a base estruturada: {type(exc).__name__}: {exc}",
                "technical": {"case_id": case_id, "error_type": type(exc).__name__, "error": str(exc), "question": question},
                "sources": {"deterministic": True, "engine": "duckdb", "table": self.TABLE},
            }

    def _answer_structured(self, case_id: str, question: str) -> dict[str, Any]:
        q = _norm(question)
        context = dict(self.memory.get(case_id) or {})

        code_match = re.search(r"\b(CHG\d{5,}|INC\d{5,})\b", question, re.IGNORECASE)
        is_detail = any(x in q for x in ["detalhe", "detalhar", "detalhes", "explique", "resuma", "resumo", "descreva", "descricao", "descrição"])
        if code_match and is_detail:
            return self._answer_detail(case_id, code_match.group(1).upper(), question)

        if "detalhe cada uma" in q or "detalhar cada uma" in q or "detalhe todas" in q or "detalhar todas" in q:
            return self._answer_bulk_detail_guard(case_id, context)

        if "quais os codigos" in q or "quais os códigos" in question.lower() or "quais são os codigos" in q or "quais são os códigos" in question.lower():
            codes = context.get("last_codes") or []
            return self._response(case_id, "\n".join(codes) if codes else "Não há códigos em memória para listar.", "list", context, {"memory_only": True, "codes": codes})

        if "aura whatsapp" in q and context.get("last_codes"):
            return self._answer_aura_whatsapp(case_id, context)

        plan = self._build_plan(question, context)
        where_sql, params = self._build_where(case_id, plan)

        monthly_answer = self._answer_monthly_kpi_if_applicable(case_id, question, q, plan)
        if monthly_answer:
            return monthly_answer

        # Intenções específicas ANTES de count/list.
        if self._is_success_rate_question(q):
            return self._answer_success_rate(case_id, where_sql, params, plan)

        if self._is_state_ranking_question(q):
            return self._answer_grouped(case_id, where_sql, params, plan, "estado", "Estados mais frequentes", "top_states")

        if self._is_group_ranking_question(q):
            return self._answer_grouped(case_id, where_sql, params, plan, "grupo_atribuicao", "Grupos mais frequentes", "top_groups")

        if self._is_distinct_question(q, "estado"):
            return self._answer_distinct(case_id, where_sql, params, plan, "estado", "Estados encontrados", "distinct_states")

        if self._is_distinct_question(q, "grupo"):
            return self._answer_distinct(case_id, where_sql, params, plan, "grupo_atribuicao", "Grupos encontrados", "distinct_groups")

        if self._is_executive_summary_question(q):
            return self._answer_executive_summary(case_id, where_sql, params, plan)

        if self._is_impact_total_question(q):
            return self._answer_impact_sum(case_id, where_sql, params, plan)

        if self._is_mttr_question(q):
            return self._answer_mttr(case_id, where_sql, params, plan)

        if self._is_major_impact_question(q):
            return self._answer_major_impact(case_id, where_sql, params, plan)

        if self._is_causes_ranking_question(q):
            return self._answer_grouped(case_id, where_sql, params, plan, "causa_origem", "Principais causas de origem", "top_causes")

        if self._is_functionality_ranking_question(q):
            return self._answer_grouped(case_id, where_sql, params, plan, "ic_impactado", "Incidentes por funcionalidade", "top_functionality")

        is_exists = self._is_exists_question(q)
        is_count = any(x in q for x in ["quantos", "quantas", "quantidade", "qtd", "qtde", "total"])

        # Removido "qual" como gatilho genérico, porque quebrava perguntas como:
        # "qual percentual..." e "qual grupo teve mais...".
        is_list = any(x in q for x in ["quais", "liste", "listar", "lista", "mostre", "codigos", "códigos"])

        if plan.get("force_list"):
            is_list = True
            is_count = False
        if plan.get("force_count"):
            is_count = True
            is_list = False

        with self._connect() as con:
            distinct_expr = "COALESCE(NULLIF(numero, ''), NULLIF(codigo_principal, ''), article_id)"

            if is_exists:
                sql = f"SELECT COUNT(DISTINCT {distinct_expr}) AS total FROM {self.TABLE} {where_sql}"
                total = int(con.execute(sql, params).fetchone()[0] or 0)
                codes_sql = f"SELECT DISTINCT {distinct_expr} AS codigo FROM {self.TABLE} {where_sql} ORDER BY codigo LIMIT 2000"
                codes = [r[0] for r in con.execute(codes_sql, params).fetchall() if r[0]]
                self._save_memory(case_id, plan, codes)
                answer = f"Sim. Encontrei {total} registro(s)." if total > 0 else "Não encontrei registros."
                return self._response(case_id, answer, "exists", plan, {"sql": sql, "params": params, "count": total, "codes_count": len(codes)})

            if is_count and not is_list:
                sql = f"SELECT COUNT(DISTINCT {distinct_expr}) AS total FROM {self.TABLE} {where_sql}"
                total = int(con.execute(sql, params).fetchone()[0] or 0)
                codes_sql = f"SELECT DISTINCT {distinct_expr} AS codigo FROM {self.TABLE} {where_sql} ORDER BY codigo LIMIT 2000"
                codes = [r[0] for r in con.execute(codes_sql, params).fetchall() if r[0]]
                self._save_memory(case_id, plan, codes)
                return self._response(case_id, str(total), "count", plan, {"sql": sql, "params": params, "count": total, "codes_count": len(codes)})

            if is_list:
                sql = f"SELECT DISTINCT {distinct_expr} AS codigo FROM {self.TABLE} {where_sql} ORDER BY codigo LIMIT 2000"
                codes = [r[0] for r in con.execute(sql, params).fetchall() if r[0]]
                self._save_memory(case_id, plan, codes)
                return self._response(case_id, "\n".join(codes) if codes else "Nenhum registro encontrado.", "list", plan, {"sql": sql, "params": params, "count": len(codes)})

        # Se a pergunta parecia analítica, mas não conseguimos interpretar com segurança,
        # não cai no RAG genérico para evitar alucinação operacional.
        return {
            "fallback_to_rag": False,
            "route": "knowledge_structured_unknown_safe",
            "query_type": "unknown_structured_intent",
            "answer_text": "Não consegui interpretar essa pergunta como consulta estruturada suportada. Tente informar período, tipo do registro, grupo, IC ou métrica desejada.",
            "summary": "Não consegui interpretar essa pergunta como consulta estruturada suportada.",
            "technical": {"case_id": case_id, "question": question, "plan": plan},
            "sources": {"deterministic": True, "engine": "duckdb", "table": self.TABLE},
        }

    def _answer_detail(self, case_id: str, code: str, question: str) -> dict[str, Any]:
        sql = f"""
            SELECT
                numero,
                codigo_tipo,
                mes,
                tipo,
                estado,
                status,
                grupo_atribuicao,
                ic_impactado,
                canal,
                categoria,
                prioridade,
                data_inicio_planejada,
                data_termino_planejada,
                article_text,
                raw_json,
                descricao,
                descricao_resumida,
                causado_pela_mudanca
            FROM {self.TABLE}
            WHERE case_id = ?
            AND numero = ?
            LIMIT 1
        """
        with self._connect() as con:
            row = con.execute(sql, [case_id, code]).fetchone()

        if not row:
            # Para não quebrar detalhamento que hoje talvez funcione melhor via RAG,
            # deixa cair no fluxo antigo.
            return {"fallback_to_rag": True, "route": "detail_not_found_fallback", "query_type": "detail"}

        cols = [
            "numero", "codigo_tipo", "mes", "tipo", "estado", "status",
            "grupo_atribuicao", "ic_impactado", "canal", "categoria",
            "prioridade", "data_inicio_planejada", "data_termino_planejada",
            "article_text", "raw_json", "descricao", "descricao_resumida", "causado_pela_mudanca",
        ]
        data = dict(zip(cols, row))
        article_text = data.get("article_text") or ""

        descricao_resumida = data.get("descricao_resumida") or _extract_field(article_text, "Descrição resumida")
        descricao = data.get("descricao") or _extract_field(article_text, "Descrição")
        tipo_teste = _extract_field(article_text, "Tipo de teste")
        plano_teste = _extract_field(article_text, "Plano de teste")
        tipo_indisp = _extract_field(article_text, "Tipo de Indisponibilidade")
        solicitacao = _extract_field(article_text, "Solicitação de")
        aberta_por = _extract_field(article_text, "Aberta por")
        atribuido = _extract_field(article_text, "Atribuído a")
        causado = data.get("causado_pela_mudanca") or _extract_field(article_text, "Causado pela mudança")
        codigo_fechamento = _extract_field(article_text, "Código de fechamento")
        anotacoes = _extract_field(article_text, "Anotações de encerramento")

        lines = [
            f"Detalhamento de {code}",
            "",
            f"- Tipo de registro: {data.get('codigo_tipo') or '-'}",
            f"- Mês: {data.get('mes') or '-'}",
            f"- Tipo: {data.get('tipo') or '-'}",
            f"- Estado/Status: {_clean_extracted_value(data.get('estado') or data.get('status') or '-')}",
            f"- IC Impactado: {data.get('ic_impactado') or '-'}",
            f"- Grupo de atribuição: {data.get('grupo_atribuicao') or '-'}",
            f"- Canal: {data.get('canal') or '-'}",
            f"- Prioridade: {data.get('prioridade') or '-'}",
            f"- Início planejado: {data.get('data_inicio_planejada') or '-'}",
            f"- Término planejado: {data.get('data_termino_planejada') or '-'}",
        ]

        if tipo_indisp:
            lines.append(f"- Tipo de indisponibilidade: {tipo_indisp}")
        if solicitacao:
            lines.append(f"- Solicitação de: {solicitacao}")
        if aberta_por:
            lines.append(f"- Aberta por: {aberta_por}")
        if atribuido:
            lines.append(f"- Atribuído a: {atribuido}")
        if causado:
            lines.append(f"- Causado pela mudança: {causado}")

        if descricao_resumida:
            lines.extend(["", "Descrição resumida:", descricao_resumida])
        if descricao:
            lines.extend(["", "Descrição:", descricao[:2500]])
        if tipo_teste:
            lines.extend(["", "Tipo de teste:", tipo_teste])
        if plano_teste:
            lines.extend(["", "Plano de teste:", plano_teste[:2500]])
        if codigo_fechamento:
            lines.extend(["", "Código de fechamento:", codigo_fechamento])
        if anotacoes:
            lines.extend(["", "Anotações de encerramento:", anotacoes[:2500]])

        memory = dict(self.memory.get(case_id) or {})
        memory["last_detail_code"] = code
        memory["last_detail_plan"] = {"codigo_tipo": data.get("codigo_tipo"), "mes": data.get("mes")}
        # Não sobrescreve listas grandes de uma pergunta anterior.
        if not memory.get("last_codes"):
            memory["last_codes"] = [code]
        self.memory[case_id] = memory

        return self._response(case_id, "\n".join(lines), "detail", {"code": code}, {"sql": sql, "record": {k: v for k, v in data.items() if k not in {"article_text", "raw_json"}}})

    def _build_plan(self, question: str, context: dict[str, Any]) -> dict[str, Any]:
        q = _norm(question)
        followup = (
            q.startswith("e ")
            or any(x in q for x in ["destes", "destas", "desses", "dessas", "deles", "delas", "essas", "esses", "foram abertas", "quais os codigos", "quais os códigos", "alguma dessas", "destes fazem", "abertas no dia", "abertos no dia"])
        )

        if followup:
            plan: dict[str, Any] = dict(context.get("last_plan") or {})
            for key in ["codigo_tipo", "mes", "dia", "is_app", "is_ecomm", "estado"]:
                if context.get(key) is not None:
                    plan[key] = context[key]
        else:
            plan = {}

        root_scope = any(x in q for x in ["quantas changes", "quantos incidentes", "e quantas changes", "e quantos incidentes"])
        if root_scope:
            plan = {}

        month = self._extract_month_from_question(q)
        if month:
            # Mês explícito representa nova intenção raiz; não deve herdar filtros do last_plan.
            plan = {"mes": month}

        if any(x in q for x in ["change", "changes", "chg", "mudanca", "mudança"]):
            plan["codigo_tipo"] = "CHG"
        if any(x in q for x in ["incidente", "incidentes"]):
            plan["codigo_tipo"] = "INC"
        if ("referencia" in q or "referência" in q or "fazem referência" in q or "causado" in q) and re.search(r"\bchg\d{5,}\b", q):
            plan["codigo_tipo"] = "INC"

        day = self._extract_day_from_question(q)
        if day:
            plan["dia"] = day
            if "aberta" in q or "abertas" in q:
                plan["codigo_tipo"] = plan.get("codigo_tipo") or "CHG"

        ic = self._extract_ic_from_question(question)
        if ic:
            plan["ic_impactado"] = ic
            # IC normalmente é pergunta de listagem, exceto quando começa com quantas/quantos.
            if not any(x in q for x in ["quantas", "quantos", "quantidade", "qtd", "qtde", "total"]):
                plan["force_list"] = True

        grupo = self._extract_group_from_question(question)
        if grupo:
            # Não remover IC se a pergunta explicitamente tiver os dois filtros.
            if "ic impactado" not in q and " ic " not in f" {q} ":
                plan.pop("ic_impactado", None)
            plan["grupo_atribuicao"] = grupo
            if any(x in q for x in ["quantas", "quantos", "quantidade", "qtd", "qtde", "total", "temos alguma", "tem alguma"]):
                plan["force_count"] = True
                plan.pop("force_list", None)

        chg_ref = re.search(r"\b(CHG\d{5,})\b", question, re.IGNORECASE)
        if chg_ref:
            plan["change_ref"] = chg_ref.group(1).upper()

        if re.search(r"\bapp\b", q) or "aplicativo" in q or "meu vivo" in q or "operacao app" in q or "operação app" in q:
            plan["is_app"] = True
        if any(x in q for x in ["ecomm", "e commerce", "e-commerce", "ecommerce"]):
            plan["is_ecomm"] = True

        estado = self._extract_state_from_question(q)
        if estado:
            plan["estado"] = estado

        tipo = self._extract_tipo_from_question(q)
        if tipo:
            plan["tipo"] = tipo

        prioridade = self._extract_priority_from_question(q)
        if prioridade:
            plan["prioridade"] = prioridade

        if self._is_systemic_question(q):
            plan["parada_sistemica"] = True

        funcionalidade = self._extract_functionality_from_question(question)
        if funcionalidade:
            plan["funcionalidade"] = funcionalidade

        return plan

    def _build_where(self, case_id: str, plan: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses = ["case_id = ?"]
        params: list[Any] = [case_id]
        codigo_tipo = plan.get("codigo_tipo")

        if codigo_tipo:
            if codigo_tipo == "CHG":
                clauses.append("""
                    (
                        numero LIKE 'CHG%'
                        OR codigo_principal LIKE 'CHG%'
                        OR codigo_tipo = 'CHG'
                        OR categoria = 'CHG'
                    )
                """)
            elif codigo_tipo == "INC":
                clauses.append("""
                    (
                        numero LIKE 'INC%'
                        OR codigo_principal LIKE 'INC%'
                        OR codigo_tipo = 'INC'
                        OR categoria = 'INC'
                    )
                """)

        if plan.get("mes"):
            clauses.append("mes = ?")
            params.append(plan["mes"])

        if plan.get("dia"):
            day_no_zero = str(int(plan["dia"]))
            day_zero = str(plan["dia"]).zfill(2)
            if codigo_tipo == "CHG":
                clauses.append("""
                    (
                        regexp_extract(COALESCE(data_inicio_planejada, ''), '20[0-9]{2}[-/][0-9]{1,2}[-/]0?([0-9]{1,2})', 1) IN (?, ?)
                        OR article_text ILIKE ?
                        OR dia IN (?, ?)
                    )
                """)
                params.extend([day_no_zero, day_zero, f"%DIA_NORMALIZADO_MARKER: {day_zero}%", day_no_zero, day_zero])
            elif codigo_tipo == "INC":
                clauses.append("""
                    (
                        regexp_extract(COALESCE(article_text, ''), 'Aberto:\\s*20[0-9]{2}[-/][0-9]{1,2}[-/]0?([0-9]{1,2})', 1) IN (?, ?)
                        OR article_text ILIKE ?
                        OR dia IN (?, ?)
                    )
                """)
                params.extend([day_no_zero, day_zero, f"%DIA_NORMALIZADO_MARKER: {day_zero}%", day_no_zero, day_zero])
            else:
                clauses.append("(article_text ILIKE ? OR dia = ?)")
                params.extend([f"%DIA_NORMALIZADO_MARKER: {day_zero}%", day_zero])

        if plan.get("ic_impactado"):
            clauses.append("ic_impactado ILIKE ?")
            params.append(f"%{plan['ic_impactado']}%")

        if plan.get("grupo_atribuicao"):
            clauses.append("grupo_atribuicao ILIKE ?")
            params.append(f"%{plan['grupo_atribuicao']}%")

        if plan.get("estado"):
            clauses.append("(estado ILIKE ? OR status ILIKE ?)")
            params.extend([f"%{plan['estado']}%", f"%{plan['estado']}%"])

        if plan.get("tipo"):
            clauses.append("tipo ILIKE ?")
            params.append(f"%{plan['tipo']}%")

        if plan.get("prioridade"):
            clauses.append("prioridade ILIKE ?")
            params.append(f"%{plan['prioridade']}%")

        if plan.get("funcionalidade"):
            clauses.append("(ic_impactado ILIKE ? OR descricao_resumida ILIKE ? OR descricao ILIKE ? OR article_text ILIKE ?)")
            func = f"%{plan['funcionalidade']}%"
            params.extend([func, func, func, func])

        if plan.get("parada_sistemica"):
            clauses.append("""
                (
                    descricao_resumida ILIKE '%INDISPONIBILIDADE%'
                    OR descricao_resumida ILIKE '%TELA DE MANUTENÇÃO%'
                    OR descricao_resumida ILIKE '%TELA DE MANUTENCAO%'
                    OR article_text ILIKE '%INDISPONIBILIDADE%'
                    OR article_text ILIKE '%TELA DE MANUTENÇÃO%'
                    OR article_text ILIKE '%TELA DE MANUTENCAO%'
                    OR raw_json ILIKE '%INDISPONIBILIDADE%'
                    OR raw_json ILIKE '%TELA DE MANUTENÇÃO%'
                    OR raw_json ILIKE '%TELA DE MANUTENCAO%'
                )
            """)

        if plan.get("is_app"):
            clauses.append("""
                (
                    canal ILIKE '%APP_MARKER%'
                    OR article_text ILIKE '%APP_MARKER: true%'
                    OR article_text ILIKE '%Canal Impactado: App Vivo%'
                    OR article_text ILIKE '%Canal impactado: APP Vivo%'
                    OR raw_json ILIKE '%"is_app": true%'
                    OR raw_json ILIKE '%"is_app":true%'
                    OR is_app = true
                )
            """)

        if plan.get("is_ecomm"):
            clauses.append("""
                (
                    canal ILIKE '%ECOMM_MARKER%'
                    OR article_text ILIKE '%ECOMM_MARKER: true%'
                    OR article_text ILIKE '%ecomm%'
                    OR article_text ILIKE '%e-commerce%'
                    OR article_text ILIKE '%ecommerce%'
                    OR raw_json ILIKE '%"is_ecomm": true%'
                    OR raw_json ILIKE '%"is_ecomm":true%'
                    OR is_ecomm = true
                )
            """)

        if plan.get("change_ref"):
            clauses.append("(article_text ILIKE ? OR raw_json ILIKE ? OR causado_pela_mudanca ILIKE ?)")
            ref = f"%CAUSADO_PELA_MUDANCA_MARKER: {plan['change_ref']}%"
            ref_json = f"%Causado pela mudança%{plan['change_ref']}%"
            ref_col = f"%{plan['change_ref']}%"
            params.extend([ref, ref_json, ref_col])

        return "WHERE " + " AND ".join(clauses), params

    def _answer_success_rate(self, case_id: str, where_sql: str, params: list[Any], plan: dict[str, Any]) -> dict[str, Any]:
        distinct_expr = "COALESCE(NULLIF(numero, ''), NULLIF(codigo_principal, ''), article_id)"
        success_condition = """
            (
                article_text ILIKE '%Código de fechamento: Bem-sucedido%'
                OR article_text ILIKE '%Codigo de fechamento: Bem-sucedido%'
                OR article_text ILIKE '%Código de fechamento: Bem sucedido%'
                OR article_text ILIKE '%Codigo de fechamento: Bem sucedido%'
                OR article_text ILIKE '%Executado com sucesso%'
                OR article_text ILIKE '%Validado por QD%'
                OR raw_json ILIKE '%Bem-sucedido%'
                OR raw_json ILIKE '%Bem sucedido%'
            )
        """

        sql = f"""
            SELECT
                COUNT(DISTINCT {distinct_expr}) AS total,
                COUNT(DISTINCT CASE WHEN {success_condition} THEN {distinct_expr} END) AS bem_sucedidas
            FROM {self.TABLE}
            {where_sql}
        """
        with self._connect() as con:
            row = con.execute(sql, params).fetchone()

        total = int(row[0] or 0)
        success = int(row[1] or 0)
        pct = round((success / total) * 100, 2) if total else 0.0

        answer = (
            f"Percentual de changes bem-sucedidas: {pct}%\n\n"
            f"- Bem-sucedidas: {success}\n"
            f"- Total analisado: {total}"
        )
        return self._response(case_id, answer, "success_rate", plan, {"sql": sql, "params": params, "total": total, "success": success, "percent": pct})

    def _answer_grouped(self, case_id: str, where_sql: str, params: list[Any], plan: dict[str, Any], column: str, title: str, query_type: str) -> dict[str, Any]:
        distinct_expr = "COALESCE(NULLIF(numero, ''), NULLIF(codigo_principal, ''), article_id)"

        # Para estado/status, higieniza valores que possam ter sido contaminados por labels seguintes,
        # como: "Encerrado(a) Data de início planejada: ..."
        if column == "estado":
            item_expr = r"regexp_replace(COALESCE(NULLIF(estado, ''), NULLIF(status, ''), '-'), '\s+Data de.*$', '')"
        else:
            item_expr = f"COALESCE(NULLIF({column}, ''), '-')"

        sql = f"""
            SELECT
                {item_expr} AS item,
                COUNT(DISTINCT {distinct_expr}) AS total
            FROM {self.TABLE}
            {where_sql}
            GROUP BY 1
            ORDER BY total DESC, item ASC
            LIMIT 20
        """
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()

        if not rows:
            return self._response(case_id, "Nenhum registro encontrado.", query_type, plan, {"sql": sql, "params": params})

        lines = [f"{title}:", ""]
        lines.extend(f"- {item}: {int(total)}" for item, total in rows)
        return self._response(case_id, "\n".join(lines), query_type, plan, {"sql": sql, "params": params, "rows": rows})

    def _answer_distinct(self, case_id: str, where_sql: str, params: list[Any], plan: dict[str, Any], column: str, title: str, query_type: str) -> dict[str, Any]:
        sql = f"""
            SELECT DISTINCT COALESCE(NULLIF({column}, ''), '-') AS item
            FROM {self.TABLE}
            {where_sql}
            ORDER BY item
            LIMIT 200
        """
        with self._connect() as con:
            rows = [r[0] for r in con.execute(sql, params).fetchall()]

        if not rows:
            return self._response(case_id, "Nenhum registro encontrado.", query_type, plan, {"sql": sql, "params": params})

        return self._response(case_id, "\n".join(rows), query_type, plan, {"sql": sql, "params": params, "count": len(rows)})

    def _answer_bulk_detail_guard(self, case_id: str, context: dict[str, Any]) -> dict[str, Any]:
        codes = context.get("last_codes") or []
        count = len(codes) or context.get("last_result_count") or 0
        sample = codes[:20]

        answer = (
            f"Encontrei {count} registro(s) no último resultado. "
            "Para não estourar o contexto, detalhe em páginas ou por código específico.\n\n"
        )
        if sample:
            answer += "Primeiros registros disponíveis:\n" + "\n".join(f"- {code}" for code in sample)
        else:
            answer += "Não há uma lista anterior em memória para detalhar."

        answer += (
            "\n\nExemplos:\n"
            "- detalhe a CHG0175923\n"
            "- detalhe os 20 primeiros\n"
            "- filtre por mês, grupo, IC ou estado antes de detalhar"
        )

        return self._response(case_id, answer, "bulk_detail_guard", context, {"memory_only": True, "codes_count": count})

    def _find_monthly_kpi_doc(self, case_id: str, plan: dict[str, Any]) -> str:
        """Busca o documento mensal de KPI/FAQ quando existir."""
        mes = plan.get("mes")
        if not mes:
            return ""
        sql = f"""
            SELECT article_text
            FROM {self.TABLE}
            WHERE case_id = ?
            AND mes = ?
            AND (
                article_text ILIKE '%KPIs do mês%'
                OR article_text ILIKE '%KPIs do mes%'
                OR article_text ILIKE '%Perguntas frequentes%'
                OR article_text ILIKE '%Resumo executivo + FAQ%'
                OR raw_json ILIKE '%KPIs do mês%'
                OR raw_json ILIKE '%Resumo executivo + FAQ%'
            )
            ORDER BY
                CASE WHEN article_text ILIKE '%Categoria: APP%' OR raw_json ILIKE '%Categoria%APP%' THEN 0 ELSE 1 END,
                LENGTH(article_text) DESC
            LIMIT 1
        """
        with self._connect() as con:
            row = con.execute(sql, [case_id, mes]).fetchone()
        return row[0] if row and row[0] else ""

    def _parse_monthly_kpi_doc(self, text: str) -> dict[str, Any]:
        if not text:
            return {}
        doc = text.replace("\r\n", "\n")
        norm_doc = _norm(doc)

        def grab(pattern: str, flags: int = re.IGNORECASE | re.DOTALL) -> str:
            m = re.search(pattern, doc, flags)
            return m.group(1).strip() if m else ""

        total = grab(r"Total de incidentes críticos[^:]*:\s*(\d+)") or grab(r"Total:\s*(\d+)\s*\(P1")
        p_match = re.search(r"P1\s*:\s*(\d+)\s*\|\s*P2\s*:\s*(\d+)\s*\|\s*P3\s*:\s*(\d+)", doc, re.IGNORECASE)
        p1 = p_match.group(1) if p_match else ""
        p2 = p_match.group(2) if p_match else ""
        p3 = p_match.group(3) if p_match else ""

        impacto_total = grab(r"Tempo total de impacto[^:]*:\s*([0-9]{1,4}:[0-9]{2}:[0-9]{2})")
        parada = grab(r"Parada sist[eê]mica[^:]*:\s*([0-9]{1,4}:[0-9]{2}:[0-9]{2})")
        mttr = grab(r"MTTR[^:]*:\s*([0-9]{1,4}:[0-9]{2}:[0-9]{2})")
        change_related = grab(r"Incidentes causados por mudan[çc]a[^:]*:\s*(\d+)")

        major = grab(r"Incidente de maior impacto:\s*(INC\d+\s*\([^\n]+?impacto\s*[0-9]{1,4}:[0-9]{2}:[0-9]{2})")
        if not major:
            major = grab(r"Qual incidente teve maior impacto[^\n]*\n\s*-\s*(INC\d+[^\n]+)")

        causes = grab(r"Principais [“\"]?Causa Origem[”\"]? \(top\):\s*(.+?)(?:\n\s*\d+\)|\n\s*10\)|\n\s*Incidentes da operação|\n\s*Incidentes por funcionalidade|\Z)")
        causes = re.sub(r"\s+", " ", causes).strip(" -") if causes else ""

        func_block = grab(r"Incidentes por funcionalidade \(Top 10\)\s*(.+?)(?:\n\s*Mês:|\Z)")
        functionality = []
        if func_block:
            for line in func_block.splitlines():
                line = line.strip(" -\t")
                if not line or ":" not in line:
                    continue
                functionality.append(line)

        incidents_block = grab(r"Incidentes da operação de APP para\s*20\d{2}-\d{2}\s*(.+?)(?:\n\s*Incidentes por funcionalidade|\Z)")
        incidents = re.findall(r"\bINC\d{5,}\b", incidents_block or "")

        return {
            "total": int(total) if str(total).isdigit() else None,
            "p1": int(p1) if str(p1).isdigit() else None,
            "p2": int(p2) if str(p2).isdigit() else None,
            "p3": int(p3) if str(p3).isdigit() else None,
            "impacto_total": impacto_total,
            "parada_sistemica": parada,
            "mttr": mttr,
            "change_related": int(change_related) if str(change_related).isdigit() else None,
            "major_impact": major,
            "causes": causes,
            "functionality": functionality,
            "incidents": incidents,
            "is_app_doc": "categoria app" in norm_doc or "operação de app" in norm_doc or "operacao de app" in norm_doc,
        }

    def _answer_monthly_kpi_if_applicable(self, case_id: str, question: str, q: str, plan: dict[str, Any]) -> dict[str, Any] | None:
        if not plan.get("mes"):
            return None

        is_kpi_question = any([
            self._is_executive_summary_question(q),
            self._is_major_impact_question(q),
            self._is_mttr_question(q),
            self._is_impact_total_question(q),
            self._is_causes_ranking_question(q),
            self._is_functionality_ranking_question(q),
            self._is_systemic_question(q),
            self._extract_priority_from_question(q) != "",
            self._is_change_related_question(q),
            "total de incidentes criticos" in q,
            "total de incidentes críticos" in q,
        ])
        if not is_kpi_question:
            return None

        doc = self._find_monthly_kpi_doc(case_id, plan)
        if not doc:
            return None
        kpi = self._parse_monthly_kpi_doc(doc)
        if not kpi:
            return None

        priority = self._extract_priority_from_question(q)
        if priority:
            value = kpi.get(priority.lower())
            if value is not None:
                return self._response(case_id, str(value), "monthly_kpi_priority", plan, {"source": "monthly_kpi_doc", "priority": priority})

        if self._is_major_impact_question(q) and kpi.get("major_impact"):
            return self._response(case_id, kpi["major_impact"], "monthly_kpi_major_impact", plan, {"source": "monthly_kpi_doc"})

        if self._is_systemic_question(q) and kpi.get("parada_sistemica"):
            return self._response(case_id, kpi["parada_sistemica"], "monthly_kpi_systemic_stop", plan, {"source": "monthly_kpi_doc"})

        if self._is_mttr_question(q) and kpi.get("mttr"):
            return self._response(case_id, kpi["mttr"], "monthly_kpi_mttr", plan, {"source": "monthly_kpi_doc"})

        if self._is_impact_total_question(q) and kpi.get("impacto_total"):
            return self._response(case_id, kpi["impacto_total"], "monthly_kpi_impact_total", plan, {"source": "monthly_kpi_doc"})

        if self._is_change_related_question(q) and kpi.get("change_related") is not None:
            return self._response(case_id, str(kpi["change_related"]), "monthly_kpi_change_related", plan, {"source": "monthly_kpi_doc"})

        if self._is_causes_ranking_question(q) and kpi.get("causes"):
            return self._response(case_id, "Principais causas de origem:\n\n" + kpi["causes"], "monthly_kpi_causes", plan, {"source": "monthly_kpi_doc"})

        if self._is_functionality_ranking_question(q) and kpi.get("functionality"):
            return self._response(case_id, "Incidentes por funcionalidade:\n\n" + "\n".join(f"- {x}" for x in kpi["functionality"]), "monthly_kpi_functionality", plan, {"source": "monthly_kpi_doc"})

        if self._is_executive_summary_question(q):
            scope = []
            if kpi.get("is_app_doc") or plan.get("is_app"):
                scope.append("APP")
            if plan.get("mes"):
                scope.append(plan["mes"])
            scope_text = " | ".join(scope) if scope else "Base analisada"
            answer = (
                f"{scope_text}: {kpi.get('total') or 0} incidentes críticos "
                f"(P1={kpi.get('p1') or 0}, P2={kpi.get('p2') or 0}, P3={kpi.get('p3') or 0}).\n"
                f"- Impacto total somado: {kpi.get('impacto_total') or '-'}.\n"
                f"- Parada sistêmica: {kpi.get('parada_sistemica') or '-'}.\n"
                f"- MTTR: {kpi.get('mttr') or '-'}.\n"
                f"- Maior impacto: {kpi.get('major_impact') or '-'}.\n"
                f"- Mudança/CHG: {kpi.get('change_related') if kpi.get('change_related') is not None else '-'} incidente(s) com indício de mudança."
            )
            return self._response(case_id, answer, "monthly_kpi_executive_summary", plan, {"source": "monthly_kpi_doc"})

        if ("total de incidentes criticos" in q or "total de incidentes críticos" in q) and kpi.get("total") is not None:
            return self._response(case_id, str(kpi["total"]), "monthly_kpi_total", plan, {"source": "monthly_kpi_doc"})

        return None

    def _answer_impact_sum(self, case_id: str, where_sql: str, params: list[Any], plan: dict[str, Any]) -> dict[str, Any]:
        sql = f"""
            SELECT
                COUNT(DISTINCT COALESCE(NULLIF(numero, ''), NULLIF(codigo_principal, ''), article_id)) AS total,
                SUM(COALESCE(tempo_impacto_segundos, 0)) AS total_seconds
            FROM {self.TABLE}
            {where_sql}
        """
        with self._connect() as con:
            row = con.execute(sql, params).fetchone()
        total = int(row[0] or 0)
        seconds = int(row[1] or 0)
        return self._response(
            case_id,
            f"Tempo total de impacto: {_seconds_to_hhmmss(seconds)}\n\n- Registros analisados: {total}",
            "impact_sum",
            plan,
            {"sql": sql, "params": params, "total": total, "seconds": seconds},
        )

    def _answer_mttr(self, case_id: str, where_sql: str, params: list[Any], plan: dict[str, Any]) -> dict[str, Any]:
        sql = f"""
            SELECT
                COUNT(DISTINCT COALESCE(NULLIF(numero, ''), NULLIF(codigo_principal, ''), article_id)) AS total,
                AVG(NULLIF(tempo_impacto_segundos, 0)) AS avg_seconds
            FROM {self.TABLE}
            {where_sql}
        """
        with self._connect() as con:
            row = con.execute(sql, params).fetchone()
        total = int(row[0] or 0)
        seconds = int(row[1] or 0) if row and row[1] is not None else 0
        return self._response(
            case_id,
            f"MTTR / tempo médio de solução: {_seconds_to_hhmmss(seconds)}\n\n- Registros analisados: {total}",
            "mttr",
            plan,
            {"sql": sql, "params": params, "total": total, "avg_seconds": seconds},
        )

    def _answer_major_impact(self, case_id: str, where_sql: str, params: list[Any], plan: dict[str, Any]) -> dict[str, Any]:
        sql = f"""
            SELECT
                COALESCE(NULLIF(numero, ''), NULLIF(codigo_principal, ''), article_id) AS codigo,
                prioridade,
                tempo_impacto,
                tempo_impacto_segundos,
                COALESCE(NULLIF(descricao_resumida, ''), NULLIF(descricao, ''), '-') AS descricao
            FROM {self.TABLE}
            {where_sql}
            ORDER BY COALESCE(tempo_impacto_segundos, 0) DESC, codigo ASC
            LIMIT 1
        """
        with self._connect() as con:
            row = con.execute(sql, params).fetchone()
        if not row:
            return self._response(case_id, "Nenhum registro encontrado.", "major_impact", plan, {"sql": sql, "params": params})
        codigo, prioridade, tempo, seconds, descricao = row
        answer = f"{codigo} ({prioridade or '-'}) — {descricao or '-'} — impacto {tempo or _seconds_to_hhmmss(seconds or 0)}"
        return self._response(case_id, answer, "major_impact", plan, {"sql": sql, "params": params, "record": row})

    def _answer_executive_summary(self, case_id: str, where_sql: str, params: list[Any], plan: dict[str, Any]) -> dict[str, Any]:
        distinct_expr = "COALESCE(NULLIF(numero, ''), NULLIF(codigo_principal, ''), article_id)"
        sql_totals = f"""
            SELECT
                COUNT(DISTINCT {distinct_expr}) AS total,
                COUNT(DISTINCT CASE WHEN prioridade ILIKE '%P1%' THEN {distinct_expr} END) AS p1,
                COUNT(DISTINCT CASE WHEN prioridade ILIKE '%P2%' THEN {distinct_expr} END) AS p2,
                COUNT(DISTINCT CASE WHEN prioridade ILIKE '%P3%' THEN {distinct_expr} END) AS p3,
                SUM(COALESCE(tempo_impacto_segundos, 0)) AS impacto_total,
                AVG(NULLIF(tempo_impacto_segundos, 0)) AS mttr,
                COUNT(DISTINCT CASE WHEN (causado_pela_mudanca <> '' OR causa_origem ILIKE '%MUDANCA%' OR causa_origem ILIKE '%MUDANÇA%') THEN {distinct_expr} END) AS mudanca
            FROM {self.TABLE}
            {where_sql}
        """
        sql_systemic = f"""
            SELECT
                COUNT(DISTINCT {distinct_expr}) AS total,
                SUM(COALESCE(tempo_impacto_segundos, 0)) AS impacto
            FROM {self.TABLE}
            {where_sql}
            AND (
                descricao_resumida ILIKE '%INDISPONIBILIDADE%'
                OR descricao_resumida ILIKE '%TELA DE MANUTENÇÃO%'
                OR descricao_resumida ILIKE '%TELA DE MANUTENCAO%'
                OR article_text ILIKE '%INDISPONIBILIDADE%'
                OR article_text ILIKE '%TELA DE MANUTENÇÃO%'
                OR article_text ILIKE '%TELA DE MANUTENCAO%'
            )
        """
        sql_top = f"""
            SELECT
                {distinct_expr} AS codigo,
                prioridade,
                tempo_impacto,
                tempo_impacto_segundos,
                COALESCE(NULLIF(descricao_resumida, ''), NULLIF(descricao, ''), '-') AS descricao
            FROM {self.TABLE}
            {where_sql}
            ORDER BY COALESCE(tempo_impacto_segundos, 0) DESC, codigo ASC
            LIMIT 1
        """
        with self._connect() as con:
            totals = con.execute(sql_totals, params).fetchone()
            systemic = con.execute(sql_systemic, params).fetchone()
            top = con.execute(sql_top, params).fetchone()
        total, p1, p2, p3, impacto, mttr, mudanca = [int(x or 0) for x in totals]
        sys_total = int(systemic[0] or 0) if systemic else 0
        sys_impact = int(systemic[1] or 0) if systemic else 0
        scope = []
        if plan.get("is_app"):
            scope.append("APP")
        if plan.get("mes"):
            scope.append(plan["mes"])
        scope_text = " | ".join(scope) if scope else "Base analisada"
        lines = [
            f"{scope_text}: {total} incidentes/registros críticos analisados (P1={p1}, P2={p2}, P3={p3}).",
            f"Impacto total somado: {_seconds_to_hhmmss(impacto)}. Parada sistêmica: {_seconds_to_hhmmss(sys_impact)} ({sys_total} registro(s)). MTTR: {_seconds_to_hhmmss(mttr)}.",
            f"Mudança/CHG: {mudanca} incidente(s) com indício de mudança.",
        ]
        if top:
            codigo, prioridade, tempo, seconds, descricao = top
            lines.append(f"Maior impacto: {codigo} ({prioridade or '-'}) — {descricao or '-'} — impacto {tempo or _seconds_to_hhmmss(seconds or 0)}.")
        return self._response(case_id, "\n".join(lines), "executive_summary", plan, {"sql_totals": sql_totals, "sql_systemic": sql_systemic, "sql_top": sql_top, "params": params})

    def _answer_aura_whatsapp(self, case_id: str, context: dict[str, Any]) -> dict[str, Any]:
        codes = context.get("last_codes") or []
        if not codes:
            return self._response(case_id, "Não", "boolean", context, {"reason": "no_last_codes"})

        placeholders = ",".join(["?"] * len(codes))
        params = [case_id] + list(codes)
        sql = f"""
            SELECT DISTINCT numero
            FROM {self.TABLE}
            WHERE case_id = ?
            AND numero IN ({placeholders})
            AND (
                ic_impactado ILIKE '%AURA WHATSAPP%'
                OR ic_impactado ILIKE '%WHATSAPP%'
                OR article_text ILIKE '%AURA WHATSAPP%'
                OR article_text ILIKE '%WHATSAPP%'
                OR raw_json ILIKE '%AURA WHATSAPP%'
                OR raw_json ILIKE '%WHATSAPP%'
            )
            ORDER BY numero
        """
        with self._connect() as con:
            found = [r[0] for r in con.execute(sql, params).fetchall() if r[0]]
        return self._response(case_id, "Sim: " + ", ".join(found) if found else "Não", "boolean", context, {"sql": sql, "found": found, "codes": codes})

    def _save_memory(self, case_id: str, plan: dict[str, Any], codes: list[str]) -> None:
        memory = dict(self.memory.get(case_id) or {})
        memory["last_plan"] = dict(plan)
        for key in ["codigo_tipo", "mes", "dia", "is_app", "is_ecomm", "estado", "tipo", "grupo_atribuicao", "ic_impactado"]:
            if key in plan:
                memory[key] = plan[key]
        if codes and len(codes) <= 100:
            memory["last_codes"] = codes
        elif codes:
            memory["last_codes"] = codes[:100]
        memory["last_result_count"] = len(codes)
        self.memory[case_id] = memory

    def _response(self, case_id: str, answer: str, query_type: str, criteria: dict[str, Any], technical: dict[str, Any]) -> dict[str, Any]:
        return {
            "fallback_to_rag": False,
            "route": "knowledge_structured_duckdb",
            "query_type": query_type,
            "answer_text": answer,
            "summary": answer,
            "technical": {"case_id": case_id, "criteria": criteria, **(technical or {})},
            "sources": {"deterministic": True, "engine": "duckdb", "table": self.TABLE},
        }

    def _normalize_month(self, value: Any) -> str:
        match = re.search(r"(20\d{2})[-/](\d{1,2})", _safe(value))
        return f"{match.group(1)}-{match.group(2).zfill(2)}" if match else ""

    def _extract_month_from_question(self, q: str) -> str:
        month_names = {
            "janeiro": "01", "jan": "01",
            "fevereiro": "02", "fev": "02",
            "marco": "03", "mar": "03",
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

        patterns = [
            r"m[eê]s\s+(\d{1,2})\s+de\s+(20\d{2})",
            r"\b(\d{1,2})\s+de\s+(20\d{2})\b",
            r"\b(20\d{2})[-/](\d{1,2})\b",
            r"\b(\d{1,2})[-/](20\d{2})\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            if pattern.startswith(r"\b(20"):
                yyyy = match.group(1)
                mm = match.group(2).zfill(2)
            elif pattern == r"\b(\d{1,2})[-/](20\d{2})\b":
                mm = match.group(1).zfill(2)
                yyyy = match.group(2)
            else:
                mm = match.group(1).zfill(2)
                yyyy = match.group(2)
            if 1 <= int(mm) <= 12:
                return f"{yyyy}-{mm}"

        for name, mm in month_names.items():
            match = re.search(rf"\b{name}\s+(?:de\s+)?(20\d{{2}})\b", q)
            if match:
                return f"{match.group(1)}-{mm}"

        return ""

    def _extract_day_from_question(self, q: str) -> str:
        patterns = [
            r"(?:dia|abertas?\s+no\s+dia|abertos?\s+no\s+dia)\s+(\d{1,2})",
            r"\b(\d{1,2})\s*$",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            day = match.group(1).zfill(2)
            if 1 <= int(day) <= 31:
                return day
        return ""

    def _extract_after(self, question: str, pattern: str) -> str:
        match = re.search(pattern, question, re.IGNORECASE)
        if not match:
            return ""
        value = match.group(1).strip().replace("?", "").strip()
        lower = value.lower()
        cut = len(value)
        for stop in ["\n", " e grupo ", " e com ", " e que ", " e quais ", " e quant", " no mes ", " no mês ", " em "]:
            idx = lower.find(stop)
            if idx >= 0:
                cut = min(cut, idx)
        return value[:cut].strip()

    def _extract_ic_from_question(self, question: str) -> str:
        explicit = self._extract_after(question, r"ic\s+impactado\s*=?\s*(.+)")
        if explicit:
            return explicit

        # Casos: "impactaram TLV_SI_ASSINE VIVO", "impactou TLV..."
        match = re.search(r"\bimpact(?:ou|aram|a|am)\s+(TLV_[A-Z0-9_\- ]+)", question, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = re.split(r"\b(?:e\s+grupo|grupo\s+de|no\s+mes|no\s+mês|em\s+\d{2}|quantas|quais)\b", value, flags=re.IGNORECASE)[0]
            return value.strip()

        match = re.search(r"\b(TLV_[A-Z0-9_\- ]+)\b", question, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = re.split(r"\b(?:e\s+grupo|grupo\s+de|no\s+mes|no\s+mês|em\s+\d{2}|quantas|quais)\b", value, flags=re.IGNORECASE)[0]
            return value.strip()

        return ""

    def _extract_group_from_question(self, question: str) -> str:
        explicit = self._extract_after(question, r"grupo\s+de\s+atribui[cç][aã]o\s*=?\s*(.+)")
        if explicit:
            return explicit

        # Casos como: "quais mudanças do grupo VIVO_DIGITAL-ECOMMERCE_PRODUCAO?"
        explicit = self._extract_after(question, r"\bgrupo\s+([A-Z0-9_][A-Z0-9_\-]+)")
        if explicit and _looks_like_assignment_group(explicit):
            return explicit

        # Casos como: "quantas VIVO_AURA_SUSTENTACAO?"
        candidates = re.findall(r"\b[A-Z0-9]+(?:_[A-Z0-9]+){1,}\b", question)
        for candidate in candidates:
            if candidate.upper().startswith("VIVO_"):
                return candidate.upper()

        return ""

    def _extract_state_from_question(self, q: str) -> str:
        if any(x in q for x in ["cancelada", "canceladas", "cancelado", "cancelados"]):
            return "Cancelado"
        if any(x in q for x in ["encerrada", "encerradas", "encerrado", "encerrados"]):
            return "Encerrado"
        if "revisao" in q or "revisão" in q:
            return "Revisão"
        return ""

    def _extract_tipo_from_question(self, q: str) -> str:
        if "emergencia" in q or "emergência" in q:
            return "Emergência"
        # Evita classificar qualquer pergunta normal como tipo Normal.
        if re.search(r"\bchanges?\s+normal\b|\bmudancas?\s+normal\b|\btipo\s+normal\b", q):
            return "Normal"
        return ""

    def _extract_priority_from_question(self, q: str) -> str:
        match = re.search(r"\b(p[123])\b", q)
        return match.group(1).upper() if match else ""

    def _extract_functionality_from_question(self, question: str) -> str:
        q = _norm(question)
        match = re.search(r"funcionalidade\s+(.+?)(?:\s+no\s+m[eê]s|\s+em\s+20\d{2}|\?|$)", question, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip('"\'')
        # Termos comuns que aparecem nas perguntas reais.
        known = [
            "checkout", "fatura", "faturas", "troca de plano", "recarga", "esim", "login",
            "ordem de servico", "ordem de serviço", "aba para voce", "aba para você",
            "pra voce", "pra você", "para voce", "para você", "vivo travel", "portabilidade",
            "seguros", "modo seguro", "vivo up", "planos", "fila de atendimento",
        ]
        for item in known:
            if item in q:
                return item
        return ""

    def _is_exists_question(self, q: str) -> bool:
        return any(x in q for x in ["temos alguma", "tem alguma", "existe alguma", "existe algum", "temos algum", "ha alguma", "há alguma"])

    def _is_systemic_question(self, q: str) -> bool:
        return any(x in q for x in ["parada sistemica", "parada sistêmica", "indisponibilidade sistemica", "indisponibilidade sistêmica", "tela de manutencao", "tela de manutenção"])

    def _is_executive_summary_question(self, q: str) -> bool:
        return ("resumo executivo" in q) or ("kpi" in q) or ("kpis" in q) or ("resumo" in q and "incidentes" in q and "principais causas" not in q)

    def _is_impact_total_question(self, q: str) -> bool:
        return any(x in q for x in ["tempo total de impacto", "impacto total somado", "tempo de impacto total"]) or ("tempo" in q and "parada" in q)

    def _is_mttr_question(self, q: str) -> bool:
        return "mttr" in q or "tempo medio de solucao" in q or "tempo médio de solução" in q or "tempo medio de solução" in q

    def _is_major_impact_question(self, q: str) -> bool:
        return "maior impacto" in q or "incidente teve maior impacto" in q or "qual incidente mais impactou" in q

    def _is_causes_ranking_question(self, q: str) -> bool:
        return ("causa" in q or "causas" in q or "causa origem" in q) and any(x in q for x in ["principais", "top", "ranking", "resumo"])

    def _is_functionality_ranking_question(self, q: str) -> bool:
        return any(x in q for x in ["incidentes por funcionalidade", "ranking de funcionalidade", "ranking por funcionalidade", "funcionalidades mais"])

    def _is_success_rate_question(self, q: str) -> bool:
        return any(x in q for x in ["percentual", "porcentagem", "taxa"]) and any(
            x in q for x in ["sucesso", "bem sucedido", "bem-sucedido", "bem sucedidas", "bem-sucedidas"]
        )

    def _is_state_ranking_question(self, q: str) -> bool:
        return any(x in q for x in ["estado", "estados", "status"]) and any(
            x in q for x in ["frequente", "frequentes", "mais comum", "mais comuns", "ranking"]
        )

    def _is_group_ranking_question(self, q: str) -> bool:
        return "grupo" in q and any(x in q for x in ["frequente", "frequentes", "mais", "ranking"])

    def _is_distinct_question(self, q: str, target: str) -> bool:
        if target == "estado":
            return any(x in q for x in ["quais estados", "listar estados", "liste os estados"])
        if target == "grupo":
            return any(x in q for x in ["quais grupos", "listar grupos", "liste os grupos"])
        return False
