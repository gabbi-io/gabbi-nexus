from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


def _safe(value: Any) -> str:
    return "" if value is None else str(value).strip()


def norm(value: Any) -> str:
    text = _safe(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
    return " ".join(text.split())


MONTHS = {
    "janeiro": "01", "jan": "01", "fevereiro": "02", "fev": "02", "marco": "03", "março": "03", "mar": "03",
    "abril": "04", "abr": "04", "maio": "05", "junho": "06", "jun": "06", "julho": "07", "jul": "07",
    "agosto": "08", "ago": "08", "setembro": "09", "set": "09", "outubro": "10", "out": "10",
    "novembro": "11", "nov": "11", "dezembro": "12", "dez": "12",
}


def normalize_month(value: Any) -> str:
    raw = _safe(value)
    m = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", raw)
    if m:
        return f"{m.group(1)}-{m.group(2).zfill(2)}"
    m = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", raw)
    if m:
        return f"{m.group(2)}-{m.group(1).zfill(2)}"
    n = norm(raw)
    m = re.search(r"\b(0?[1-9]|1[0-2])\s+de\s+(20\d{2})\b", n)
    if m:
        return f"{m.group(2)}-{m.group(1).zfill(2)}"
    y = re.search(r"\b(20\d{2})\b", n)
    if y:
        for name, mm in MONTHS.items():
            if re.search(rf"\b{re.escape(norm(name))}\b", n):
                return f"{y.group(1)}-{mm}"
    return ""


def _extract_label(text: str, labels: list[str], stop_labels: list[str] | None = None) -> str:
    # Captura valores em linhas no formato "Label: valor" até a próxima quebra ou próximo label conhecido.
    escaped = [re.escape(l) for l in labels]
    label_re = r"(?:" + "|".join(escaped) + r")"
    stop = stop_labels or [
        "Mês", "Mes", "Número", "Numero", "Prioridade", "Estado", "Status", "Tipo", "Aberto", "Resolvido", "Encerrado",
        "Data de início planejada", "Data de inicio planejada", "Data de término planejada", "Data de termino planejada",
        "IC Impactado", "Grupo de atribuição", "Grupo de atribuicao", "Canal impactado", "Causado pela mudança",
        "Descrição resumida", "Descricao resumida", "Descrição", "Descricao", "Causa provável", "Causa Origem", "Serviço", "Servico"
    ]
    stop_re = r"(?:" + "|".join(re.escape(s) for s in stop) + r")\s*:"
    pattern = rf"(?is)(?:^|\n|\r)\s*{label_re}\s*:\s*(.*?)(?=\n\s*{stop_re}|\r\n\s*{stop_re}|$)"
    m = re.search(pattern, text or "")
    if not m:
        return ""
    value = m.group(1).strip()
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ;,.-")


def _as_bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    n = norm(value)
    if n in {"1", "true", "sim", "yes", "y"}:
        return "true"
    if n in {"0", "false", "nao", "não", "no", "n"}:
        return "false"
    return ""


def _extract_json_document(article_document: Any) -> dict[str, Any]:
    if article_document is None:
        return {}
    if isinstance(article_document, dict):
        return dict(article_document)
    text = _safe(article_document)
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        return {}


def _pick(data: dict[str, Any], *keys: str) -> str:
    # Busca exata e depois case-insensitive.
    for key in keys:
        if key in data and _safe(data.get(key)):
            return _safe(data.get(key))
    lowered = {norm(k): v for k, v in data.items()}
    for key in keys:
        nk = norm(key)
        if nk in lowered and _safe(lowered[nk]):
            return _safe(lowered[nk])
    return ""


def parse_article_to_structured_row(
    *,
    article_text: str,
    article_document: Any = None,
    metadata: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Converte um artigo Gabbi de INC/CHG em linha canônica para DuckDB.

    Esta função é intencionalmente determinística. Ela não usa LLM.
    O objetivo é impedir que perguntas analíticas sejam respondidas por RAG/top-k.
    """
    text = _safe(article_text)
    metadata = metadata or {}
    defaults = defaults or {}
    doc = _extract_json_document(article_document)
    combined = "\n".join([text, json.dumps(doc, ensure_ascii=False, default=str) if doc else ""])

    numero = _pick(doc, "Número", "Numero", "numero", "number", "codigo", "Código") or _extract_label(text, ["Número", "Numero"])
    if not numero:
        m = re.search(r"\b((?:CHG|INC)\d{5,})\b", combined, flags=re.IGNORECASE)
        numero = m.group(1).upper() if m else ""

    codigo_tipo = _pick(doc, "codigo_tipo", "Código Tipo", "Codigo Tipo") or _safe(defaults.get("codigo_tipo"))
    if not codigo_tipo:
        if re.match(r"^CHG\d+", numero, re.IGNORECASE):
            codigo_tipo = "CHG"
        elif re.match(r"^INC\d+", numero, re.IGNORECASE):
            codigo_tipo = "INC"
        elif re.search(r"\bCHG\d{5,}\b", combined, re.IGNORECASE):
            codigo_tipo = "CHG"
        elif re.search(r"\bINC\d{5,}\b", combined, re.IGNORECASE):
            codigo_tipo = "INC"

    row = {
        **defaults,
        "article_id": _safe(metadata.get("article_id") or defaults.get("article_id")),
        "article_ref_id": _safe(metadata.get("ref_id") or metadata.get("article_ref_id") or defaults.get("article_ref_id")),
        "topic_id": _safe(metadata.get("topic_id") or defaults.get("topic_id")),
        "topic_name": _safe(metadata.get("topic_name") or defaults.get("topic_name")),
        "topic_description": _safe(metadata.get("topic_description") or defaults.get("topic_description")),
        "source_id": _safe(metadata.get("source_id") or metadata.get("article_id") or defaults.get("source_id")),
        "codigo_tipo": codigo_tipo.upper(),
        "codigo_principal": numero.upper(),
        "numero": numero.upper(),
        "mes": _pick(doc, "Mês", "Mes", "mes") or _extract_label(text, ["Mês", "Mes"]) or normalize_month(combined),
        "tipo": _pick(doc, "Tipo", "tipo") or _extract_label(text, ["Tipo"]),
        "estado": _pick(doc, "Estado", "estado") or _extract_label(text, ["Estado"]),
        "status": _pick(doc, "Status", "status") or _extract_label(text, ["Status"]),
        "grupo_atribuicao": _pick(doc, "Grupo de atribuição", "Grupo de atribuicao", "grupo_atribuicao", "assignment_group") or _extract_label(text, ["Grupo de atribuição", "Grupo de atribuicao"]),
        "ic_impactado": _pick(doc, "IC Impactado", "IC impactado", "ic_impactado", "CMDB CI") or _extract_label(text, ["IC Impactado", "IC impactado"]),
        "canal": _pick(doc, "Canal impactado", "Canal Impactado", "canal", "Canal") or _extract_label(text, ["Canal impactado", "Canal Impactado", "Canal"]),
        "categoria": _pick(doc, "Categoria", "categoria", "category") or _extract_label(text, ["Categoria"]),
        "prioridade": _pick(doc, "Prioridade", "prioridade", "priority") or _extract_label(text, ["Prioridade"]),
        "data_inicio_planejada": _pick(doc, "Data de início planejada", "Data de inicio planejada", "data_inicio_planejada") or _extract_label(text, ["Data de início planejada", "Data de inicio planejada"]),
        "data_termino_planejada": _pick(doc, "Data de término planejada", "Data de termino planejada", "data_termino_planejada") or _extract_label(text, ["Data de término planejada", "Data de termino planejada"]),
        "aberto": _pick(doc, "Aberto", "aberto", "opened_at") or _extract_label(text, ["Aberto"]),
        "resolvido": _pick(doc, "Resolvido", "resolvido") or _extract_label(text, ["Resolvido"]),
        "encerrado": _pick(doc, "Encerrado", "encerrado") or _extract_label(text, ["Encerrado"]),
        "causado_pela_mudanca": _pick(doc, "Causado pela mudança", "Causado pela mudanca", "causado_pela_mudanca") or _extract_label(text, ["Causado pela mudança", "Causado pela mudanca"]),
        "servico": _pick(doc, "Serviço", "Servico", "servico", "Serviço afetado") or _extract_label(text, ["Serviço", "Servico"]),
        "descricao_resumida": _pick(doc, "Descrição resumida", "Descricao resumida", "descricao_resumida") or _extract_label(text, ["Descrição resumida", "Descricao resumida"]),
        "article_text": text,
        "raw_json": json.dumps({"document": doc, "metadata": metadata}, ensure_ascii=False, default=str),
    }

    if not row["mes"]:
        row["mes"] = normalize_month(row.get("aberto") or row.get("data_inicio_planejada") or combined)

    signal = norm(" ".join([row.get("canal", ""), row.get("descricao_resumida", ""), row.get("article_text", "")]))
    row["is_app"] = _as_bool_text(_pick(doc, "is_app")) or ("true" if re.search(r"\bapp\b|aplicativo vivo|app vivo", signal) else "false")
    row["is_ecomm"] = _as_bool_text(_pick(doc, "is_ecomm")) or ("true" if re.search(r"ecomm|e-commerce|ecommerce|loja online|portal vivo|hybris", signal) else "false")

    if not row.get("codigo_tipo"):
        return {}
    if not row.get("numero"):
        return {}
    return row
