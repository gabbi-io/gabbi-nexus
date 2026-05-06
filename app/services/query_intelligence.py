from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any


def _norm(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9_:/\-\s\.]+", " ", text)
    return " ".join(text.split())


def _compact(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


@dataclass
class QueryFilter:
    column: str
    operator: str
    value: Any
    source: str = "current"
    confidence: float = 1.0

    def key(self) -> tuple[str, str]:
        return (self.column, str(self.value).lower())


@dataclass
class QueryPlan:
    use_tabular: bool
    intent: str = "list"
    filters: list[QueryFilter] = field(default_factory=list)
    group_by: str | None = None
    limit: int | None = 20
    followup: bool = False
    reset_context: bool = False
    reason: str = ""
    explicit_fields: list[str] = field(default_factory=list)
    subject_tokens: list[str] = field(default_factory=list)


class QueryIntelligenceService:
    QUANT_MARKERS = {"quantos", "quantas", "quantidade", "numero", "número", "total", "conta", "contar", "qtd", "qtde"}
    LIST_MARKERS = {"liste", "listar", "lista", "quais", "mostre", "detalhe", "detalhar", "descreva", "descrever", "relacione", "traga", "exiba"}
    FOLLOWUP_MARKERS = {"eles", "elas", "deles", "delas", "dessas", "desses", "destas", "destes", "cada uma", "cada um", "os registros", "essas changes", "esses chamados"}
    SOFT_FOLLOWUP_PREFIXES = ("e ", "e com ", "e quant", "e quais", "e no ", "e na ", "e em ")
    BAD_FILTER_VALUES = {"qual", "quais", "quanto", "quantos", "quantas", "total", "tipo", "me", "o", "a", "os", "as", "de", "do", "da", "dos", "das", "sobre", "contexto", "base", "conhecimento", "registro", "registros", "cada", "um", "uma", "eles", "elas", "com", "por", "temos", "existe", "existem", "fale", "explique", "descreva", "liste", "e", "no", "na", "em"}

    FIELD_ALIASES = {
        "grupo_atribuicao": ["grupo de atribuicao", "grupo de atribuição", "assignment group", "grupo atribuição", "grupo atribuicao", "grupo"],
        "ic_impactado": ["ic impactado", "ic", "ci impactado", "item de configuracao", "item de configuração", "configuration item", "cmdb ci"],
        "codigo_tipo": ["codigo tipo", "código tipo", "tipo codigo", "tipo de codigo", "tipo de código", "tipo do codigo", "tipo do código"],
        "tipo": ["tipo", "tipo de change", "tipo da change", "tipo do registro", "tipo de registro"],
        "estado": ["estado", "state"],
        "status": ["status", "situacao", "situação"],
        "prioridade": ["prioridade", "priority", "criticidade", "severidade"],
        "mes": ["mes", "mês", "competencia", "competência", "periodo", "período"],
        "numero": ["numero", "número", "number", "id", "codigo", "código"],
        "categoria": ["categoria", "category"],
        "canal": ["canal", "channel"],
    }
    TYPE_SYNONYMS = {
        "CHG": ["chg", "change", "changes", "mudanca", "mudança", "mudancas", "mudanças", "alteracao", "alteração"],
        "INC": ["inc", "incidente", "incidentes", "chamado", "chamados", "ocorrencia", "ocorrência", "ocorrencias", "ocorrências"],
    }
    MONTHS = {"janeiro": "01", "jan": "01", "fevereiro": "02", "fev": "02", "marco": "03", "março": "03", "mar": "03", "abril": "04", "abr": "04", "maio": "05", "mai": "05", "junho": "06", "jun": "06", "julho": "07", "jul": "07", "agosto": "08", "ago": "08", "setembro": "09", "set": "09", "outubro": "10", "out": "10", "novembro": "11", "nov": "11", "dezembro": "12", "dez": "12"}

    def build_plan(self, question: str, columns: list[str], previous_plan: dict[str, Any] | None = None) -> QueryPlan:
        q_norm = _norm(question)
        intent = self._detect_intent(q_norm)
        explicit_filters = self._extract_explicit_field_filters(question, columns)
        code_filters = self._extract_code_type_filters(q_norm, columns)
        date_filters = self._extract_month_filters(q_norm, columns)
        generic_filters = self._extract_safe_generic_filters(q_norm, columns)
        current_filters = self._dedupe_filters(explicit_filters + code_filters + date_filters + generic_filters)
        followup = self._looks_like_followup(q_norm, bool(explicit_filters or date_filters or code_filters))
        reset_context = self._looks_like_reset(q_norm)
        inherited: list[QueryFilter] = []
        if followup and previous_plan and not reset_context:
            inherited = self._compatible_previous_filters(previous_plan, current_filters)
        current_entity_cols = {f.column for f in current_filters if f.column in {"codigo_tipo", "numero", "codigo_principal"}}
        if current_entity_cols:
            inherited = [f for f in inherited if f.column not in current_entity_cols]
        filters = self._dedupe_filters(inherited + current_filters)
        use_tabular = bool(filters) or intent in {"count", "list", "group"} or self._has_known_tabular_terms(q_norm, columns)
        if intent == "describe" and not filters and not followup:
            use_tabular = False
        group_by = self._extract_group_by(q_norm, columns) if intent == "group" else None
        limit = 150 if intent in {"list", "describe"} else 20
        return QueryPlan(use_tabular=use_tabular, intent=intent, filters=filters, group_by=group_by, limit=limit, followup=followup, reset_context=reset_context, reason="query_intelligence_v3", explicit_fields=[f.column for f in explicit_filters], subject_tokens=self._subject_tokens(filters))

    def _detect_intent(self, q_norm: str) -> str:
        tokens = set(q_norm.split())
        if tokens & self.QUANT_MARKERS:
            return "count"
        if any(m in q_norm for m in ["agrupe", "agrupar", "distribuicao", "distribuição", " por "]):
            return "group"
        if tokens & self.LIST_MARKERS:
            return "list"
        if any(x in q_norm for x in ["sobre o que", "contexto", "base de conhecimento", "o que se trata"]):
            return "describe"
        return "describe"

    def _looks_like_followup(self, q_norm: str, has_current_filters: bool) -> bool:
        if any(marker in q_norm for marker in self.FOLLOWUP_MARKERS):
            return True
        if q_norm.startswith(self.SOFT_FOLLOWUP_PREFIXES) and has_current_filters:
            return True
        return False

    def _looks_like_reset(self, q_norm: str) -> bool:
        return any(m in q_norm for m in ["novo assunto", "mudando de assunto", "agora me fale", "agora sobre", "contexto geral", "base de conhecimento", "sobre o que se trata", "limpa", "zerar", "sem considerar"])

    def _has_known_tabular_terms(self, q_norm: str, columns: list[str]) -> bool:
        joined_cols = " ".join(_norm(c) for c in columns)
        return any(token in joined_cols for token in q_norm.split() if len(token) > 4)

    def _resolve_column(self, raw_field: str, columns: list[str]) -> str | None:
        raw_norm = _norm(raw_field)
        compact_raw = _compact(raw_field)
        candidates: list[tuple[str, float]] = []
        for col in columns:
            col_norm = _norm(col)
            score = SequenceMatcher(None, raw_norm, col_norm).ratio()
            if compact_raw and compact_raw == _compact(col): score += 1.0
            elif compact_raw and compact_raw in _compact(col): score += 0.55
            candidates.append((col, score))
        for canonical, aliases in self.FIELD_ALIASES.items():
            if raw_norm == canonical or raw_norm in [_norm(a) for a in aliases]:
                for col in columns:
                    if _compact(canonical) == _compact(col) or _compact(canonical) in _compact(col): candidates.append((col, 2.0))
                    else:
                        for alias in aliases:
                            if _compact(alias) == _compact(col) or _compact(alias) in _compact(col): candidates.append((col, 1.8))
        if not candidates: return None
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0] if candidates[0][1] >= 0.55 else None

    def _extract_explicit_field_filters(self, question: str, columns: list[str]) -> list[QueryFilter]:
        filters: list[QueryFilter] = []
        pattern = re.compile(r"(?P<field>[A-Za-zÀ-ÿ0-9 _/\-]{2,60})\s*[:=]\s*(?P<value>.+?)(?=(?:\s+\be\s+[A-Za-zÀ-ÿ0-9 _/\-]{2,60}\s*[:=])|[\n\r]|$)", flags=re.IGNORECASE)
        for match in pattern.finditer(question):
            raw_field = match.group("field").strip()
            raw_value = match.group("value").strip().strip(".;, ")
            raw_value = re.sub(r"[”\"']+$", "", raw_value).strip()
            col = self._resolve_column(raw_field, columns)
            if not col or not self._valid_filter_value(raw_value): continue
            op = "eq" if col in {"mes", "tipo", "estado", "status", "prioridade", "codigo_tipo"} else "contains"
            filters.append(QueryFilter(column=col, operator=op, value=raw_value, source="explicit", confidence=1.0))
        return filters

    def _extract_code_type_filters(self, q_norm: str, columns: list[str]) -> list[QueryFilter]:
        filters: list[QueryFilter] = []
        codigo_col = self._resolve_column("codigo_tipo", columns)
        if codigo_col:
            for code, synonyms in self.TYPE_SYNONYMS.items():
                if any(re.search(rf"\b{re.escape(_norm(syn))}\b", q_norm) for syn in synonyms):
                    filters.append(QueryFilter(codigo_col, "contains", code, "semantic_type", 0.95)); break
        number_col = self._resolve_column("numero", columns) or self._resolve_column("codigo_principal", columns)
        principal_col = self._resolve_column("codigo_principal", columns)
        for code in re.findall(r"\b(?:CHG|INC|REQ|RITM|TASK|Z[A-Z0-9]{2,})[A-Z0-9\-]*\d+\b", q_norm, flags=re.IGNORECASE):
            target_col = principal_col or number_col
            if target_col: filters.append(QueryFilter(target_col, "contains", code.upper(), "code", 1.0))
        return filters

    def _extract_month_filters(self, q_norm: str, columns: list[str]) -> list[QueryFilter]:
        mes_col = self._resolve_column("mes", columns)
        if not mes_col: return []
        filters: list[QueryFilter] = []
        m = re.search(r"\b(?:chg|inc|mes|m[eê]s)?\s*:?\s*(0?[1-9]|1[0-2])[-/](20\d{2})\b", q_norm)
        if m: filters.append(QueryFilter(mes_col, "eq", f"{m.group(2)}-{m.group(1).zfill(2)}", "month", 0.95))
        for yyyy, mm in re.findall(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", q_norm):
            filters.append(QueryFilter(mes_col, "eq", f"{yyyy}-{mm.zfill(2)}", "month", 0.95))
        for name, mm in self.MONTHS.items():
            if re.search(rf"\b{name}\b", q_norm):
                y = re.search(r"\b(20\d{2})\b", q_norm)
                if y: filters.append(QueryFilter(mes_col, "eq", f"{y.group(1)}-{mm}", "month_name", 0.9))
        return filters

    def _extract_safe_generic_filters(self, q_norm: str, columns: list[str]) -> list[QueryFilter]:
        filters: list[QueryFilter] = []
        tipo_col = self._resolve_column("tipo", columns)
        if tipo_col:
            m = re.search(r"\b(?:do|da|de)?\s*tipo\s+([a-z0-9_\-]+)\b", q_norm)
            if m and self._valid_filter_value(m.group(1)):
                filters.append(QueryFilter(tipo_col, "eq", m.group(1).capitalize(), "generic_type", 0.75))
        prio_col = self._resolve_column("prioridade", columns)
        if prio_col:
            m = re.search(r"\b(P[0-9])\b", q_norm, flags=re.IGNORECASE)
            if m: filters.append(QueryFilter(prio_col, "contains", m.group(1).upper(), "priority", 0.9))
        return filters

    def _extract_group_by(self, q_norm: str, columns: list[str]) -> str | None:
        if " por " not in q_norm: return None
        tail = re.split(r"[\?\.,;]", q_norm.split(" por ", 1)[1])[0].strip()
        return self._resolve_column(tail, columns)

    def _compatible_previous_filters(self, previous_plan: dict[str, Any], current_filters: list[QueryFilter]) -> list[QueryFilter]:
        raw_filters = previous_plan.get("filters") or []
        current_cols = {f.column for f in current_filters}
        inherited: list[QueryFilter] = []
        for f in raw_filters:
            if not isinstance(f, dict): continue
            col, val = f.get("column"), f.get("value")
            if not col or col in current_cols or not self._valid_filter_value(val): continue
            if col in {"codigo_tipo", "mes", "tipo", "estado", "status", "prioridade", "grupo_atribuicao", "ic_impactado", "canal", "categoria"}:
                inherited.append(QueryFilter(col, f.get("operator", "contains"), val, "inherited", 0.8))
        return inherited[:3]

    def _valid_filter_value(self, value: Any) -> bool:
        if value is None: return False
        text = _norm(value)
        if not text or text in self.BAD_FILTER_VALUES: return False
        if len(text) <= 1: return False
        if len(text) <= 3 and not re.fullmatch(r"p\d|inc|chg|\d+", text): return False
        if any(bad in text.split() for bad in self.BAD_FILTER_VALUES) and len(text.split()) <= 3: return False
        return True

    def _dedupe_filters(self, filters: list[QueryFilter]) -> list[QueryFilter]:
        out, seen = [], set()
        for f in filters:
            if not self._valid_filter_value(f.value): continue
            key = f.key()
            if key in seen: continue
            seen.add(key); out.append(f)
        return out

    def _subject_tokens(self, filters: list[QueryFilter]) -> list[str]:
        return [f"{f.column}:{f.value}" for f in filters if f.column in {"codigo_tipo", "mes", "tipo", "grupo_atribuicao", "ic_impactado", "numero", "codigo_principal"}]

    @staticmethod
    def plan_to_dict(plan: QueryPlan) -> dict[str, Any]:
        return {"use_tabular": plan.use_tabular, "intent": plan.intent, "filters": [f.__dict__ for f in plan.filters], "group_by": plan.group_by, "limit": plan.limit, "followup": plan.followup, "reset_context": plan.reset_context, "reason": plan.reason, "explicit_fields": plan.explicit_fields, "subject_tokens": plan.subject_tokens}
