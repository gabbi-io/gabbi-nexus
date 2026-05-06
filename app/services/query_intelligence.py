from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Iterable


@dataclass
class QueryPlan:
    use_tabular: bool
    intent: str = "document"
    filters: list[dict[str, Any]] = field(default_factory=list)
    group_by: str | None = None
    limit: int | None = None
    followup: bool = False
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_tabular": self.use_tabular,
            "intent": self.intent,
            "filters": self.filters,
            "group_by": self.group_by,
            "limit": self.limit,
            "followup": self.followup,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class QueryIntelligence:
    """Camada genérica de interpretação de consulta.

    Objetivo:
    - Manter RAG/Nexus como rota principal para perguntas abertas.
    - Usar CSV/tabular apenas quando a pergunta pedir contagem, listagem, agrupamento
      ou quando for um follow-up real de uma consulta tabular anterior.
    - Evitar filtros acidentais como `categoria contains qual/tipo/contexto`.
    - Interpretar sinônimos de forma controlada e baseada na estrutura da própria base.
    """

    STOP_FILTER_VALUES = {
        "qual", "quais", "quanto", "quantos", "quantas", "total", "contagem", "quantidade",
        "me", "liste", "listar", "mostre", "mostra", "descreva", "descrever", "detalhe", "detalhar",
        "eles", "elas", "deles", "delas", "essas", "esses", "estas", "estes", "cada", "uma", "um",
        "base", "contexto", "assunto", "dados", "arquivo", "documento", "documentos", "artigos",
        "tipo", "com", "de", "da", "do", "das", "dos", "em", "para", "sobre", "entao", "então", "e",
    }

    QUANT_MARKERS = {
        "quantos", "quantas", "quanto", "quantidade", "total", "contar", "contagem", "número", "numero",
    }
    LIST_MARKERS = {"liste", "listar", "lista", "mostre", "mostrar", "quais", "descreva", "detalhe", "detalhar"}
    GROUP_MARKERS = {"por", "agrup", "distribuição", "distribuicao", "ranking", "total por"}
    FOLLOWUP_MARKERS = {
        "eles", "elas", "deles", "delas", "dessas", "desses", "destas", "destes",
        "cada uma", "cada um", "essas", "esses", "as mesmas", "os mesmos", "delas", "deles",
    }
    RESET_MARKERS = {
        "contexto da base", "contexto dessa base", "sobre o que", "o que se trata", "descreva a base",
        "explique a base", "me explique a base", "resumo da base", "visão geral", "visao geral",
    }

    SEMANTIC_TYPE_SYNONYMS = {
        "CHG": ["chg", "change", "changes", "mudança", "mudancas", "mudanças", "alteração", "alteracoes", "alterações"],
        "INC": ["inc", "incidente", "incidentes", "chamado", "chamados", "ocorrência", "ocorrencias", "ocorrências"],
    }

    FIELD_ALIASES = {
        "codigo_tipo": ["codigo_tipo", "tipo de código", "tipo codigo", "código tipo", "codigo", "categoria do codigo"],
        "numero": ["numero", "número", "registro", "id", "chamado", "ticket", "change", "incidente"],
        "codigo_principal": ["codigo_principal", "código principal", "codigo principal", "chg", "inc"],
        "mes": ["mes", "mês", "competencia", "competência", "periodo", "período"],
        "tipo": ["tipo", "tipo da change", "tipo do registro", "classe", "natureza"],
        "categoria": ["categoria", "categoria do registro", "tema"],
        "canal": ["canal", "canal impactado"],
        "estado": ["estado", "situação", "situacao"],
        "status": ["status"],
        "prioridade": ["prioridade", "criticidade", "severidade"],
        "grupo_atribuicao": ["grupo de atribuição", "grupo atribuicao", "grupo_atribuicao", "grupo", "assignment group", "grupo responsável", "grupo responsavel"],
        "ic_impactado": ["ic impactado", "ic", "item de configuração", "item de configuracao", "cmdb", "ci impactado"],
        "project_name": ["projeto", "project"],
        "topic_name": ["topico", "tópico", "assunto"],
        "article_text": ["texto", "conteudo", "conteúdo", "descrição", "descricao"],
    }

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

    def normalize(self, text: str) -> str:
        text = (text or "").lower().strip()
        repl = {
            "ç": "c", "ã": "a", "á": "a", "à": "a", "â": "a", "ä": "a",
            "é": "e", "ê": "e", "è": "e", "í": "i", "ó": "o", "ô": "o", "õ": "o", "ú": "u",
        }
        for k, v in repl.items():
            text = text.replace(k, v)
        return " ".join(re.sub(r"[^a-z0-9_:\-/\.]+", " ", text).split())

    def plan(self, question: str, columns: list[str], previous_context: dict[str, Any] | None = None) -> dict[str, Any]:
        q_raw = question or ""
        q = self.normalize(q_raw)
        previous_context = previous_context or {}

        if self._is_reset_question(q):
            return QueryPlan(False, intent="document", confidence=0.95, reason="reset_or_general_context_question").to_dict()

        explicit_filters = self._extract_explicit_field_filters(q_raw, columns)
        code_filters = self._extract_codes(q, columns)
        semantic_filters = self._extract_semantic_type_filters(q, columns)
        date_filters = self._extract_month_filters(q, columns)
        type_filters = self._extract_type_filters(q_raw, columns)
        priority_filters = self._extract_priority_filters(q, columns)

        new_filters = self._dedupe_filters(explicit_filters + code_filters + semantic_filters + date_filters + type_filters + priority_filters)
        intent = self._detect_intent(q)
        is_followup = self._is_followup(q, bool(new_filters), previous_context)

        filters: list[dict[str, Any]] = []
        if is_followup:
            filters.extend(self._safe_previous_filters(previous_context, new_filters))
        filters.extend(new_filters)
        filters = self._dedupe_filters(filters)

        group_by = self._detect_group_by(q, columns)
        if group_by:
            intent = "group"

        # Regra central: só usa tabular quando há necessidade analítica/listagem clara,
        # filtro explícito/código, ou follow-up real de consulta tabular anterior.
        has_tabular_signal = (
            intent in {"count", "group", "list"}
            or bool(filters)
            or is_followup
        )
        if not has_tabular_signal:
            return QueryPlan(False, intent="document", confidence=0.2, reason="no_tabular_signal").to_dict()

        # Evita que perguntas abertas virem consulta tabular só porque contêm palavras comuns.
        if intent == "document" and not filters and not is_followup:
            return QueryPlan(False, intent="document", confidence=0.3, reason="open_question_without_filters").to_dict()

        confidence = 0.55
        if filters:
            confidence += 0.2
        if explicit_filters:
            confidence += 0.15
        if is_followup:
            confidence += 0.1
        if intent in {"count", "group"}:
            confidence += 0.1

        if intent == "document":
            intent = "list" if (filters or is_followup) else "document"

        return QueryPlan(
            use_tabular=True,
            intent=intent,
            filters=filters,
            group_by=group_by,
            limit=150 if intent in {"list", "detail"} else 20,
            followup=is_followup,
            confidence=min(confidence, 0.98),
            reason="tabular_signal_detected",
        ).to_dict()

    def _is_reset_question(self, q: str) -> bool:
        return any(marker in q for marker in self.RESET_MARKERS)

    def _detect_intent(self, q: str) -> str:
        if any(marker in q for marker in self.QUANT_MARKERS):
            if any(marker in q for marker in [" por ", "agrup", "distribuicao", "distribuicao", "ranking", "total por"]):
                return "group"
            return "count"
        if any(marker in q for marker in self.LIST_MARKERS):
            return "list"
        if any(marker in q for marker in self.GROUP_MARKERS) and any(x in q for x in ["total", "quant", "cont"]):
            return "group"
        return "document"

    def _is_followup(self, q: str, has_new_filters: bool, previous_context: dict[str, Any]) -> bool:
        if not previous_context or not previous_context.get("filters"):
            return False
        if previous_context.get("expired"):
            return False
        if q.startswith("e ") or q.startswith("com ") or q.startswith("tambem ") or q.startswith("também "):
            return True
        if any(marker in q for marker in self.FOLLOWUP_MARKERS):
            return True
        # Se houver novo filtro explícito e a frase for curta, é provável refinamento.
        if has_new_filters and len(q.split()) <= 10 and not self._is_reset_question(q):
            return True
        return False

    def _safe_previous_filters(self, previous_context: dict[str, Any], new_filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previous = previous_context.get("filters") or []
        # Se a pergunta nova muda a entidade principal (ex.: INC -> CHG), remove filtros de codigo_tipo anteriores.
        new_code_type = {str(f.get("value", "")).upper() for f in new_filters if f.get("column") == "codigo_tipo"}
        out = []
        for filt in previous:
            if not self._is_valid_filter(filt):
                continue
            if filt.get("column") == "codigo_tipo" and new_code_type and str(filt.get("value", "")).upper() not in new_code_type:
                continue
            out.append(dict(filt))
        return out

    def _extract_explicit_field_filters(self, question: str, columns: list[str]) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        # Captura padrões universais: "Campo: Valor" ou "Campo = Valor".
        pattern = r"([A-Za-zÀ-ÿ0-9_\s]{2,45})\s*(?:[:=])\s*([^\n\r;,]+)"
        for raw_field, raw_value in re.findall(pattern, question or ""):
            field = raw_field.strip()
            value = raw_value.strip().strip('"\'')
            col = self.resolve_column(columns, [field])
            if col and self._valid_value(value):
                op = "eq" if col in {"mes", "tipo", "estado", "status", "prioridade", "codigo_tipo"} else "contains"
                filters.append({"column": col, "operator": op, "value": value})
        return filters

    def _extract_codes(self, q: str, columns: list[str]) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        # CHG:10-2025 ou INC:07-2025 como tópico/recorte.
        topic_month = re.search(r"\b(chg|inc)\s*[:\-]\s*(\d{1,2})[\-/](\d{4})\b", q)
        if topic_month:
            code = topic_month.group(1).upper()
            month = topic_month.group(2).zfill(2)
            year = topic_month.group(3)
            code_col = self.resolve_column(columns, ["codigo_tipo"])
            mes_col = self.resolve_column(columns, ["mes"])
            if code_col:
                filters.append({"column": code_col, "operator": "contains", "value": code})
            if mes_col:
                filters.append({"column": mes_col, "operator": "eq", "value": f"{year}-{month}"})
            return filters

        codes = re.findall(r"\b(CHG|INC|REQ|RITM|TASK|Z[A-Z]{1,4})\s*0*([0-9]{2,10})\b", q.upper())
        if codes:
            number_col = self.resolve_column(columns, ["numero", "codigo_principal", "identificadores", "article_text"])
            code_type_col = self.resolve_column(columns, ["codigo_tipo"])
            for prefix, digits in codes:
                full = f"{prefix}{digits}"
                if number_col:
                    filters.append({"column": number_col, "operator": "contains", "value": full})
                if code_type_col and prefix in {"CHG", "INC"}:
                    filters.append({"column": code_type_col, "operator": "contains", "value": prefix})
        return filters

    def _extract_semantic_type_filters(self, q: str, columns: list[str]) -> list[dict[str, Any]]:
        col = self.resolve_column(columns, ["codigo_tipo"])
        if not col:
            return []
        filters = []
        for code, aliases in self.SEMANTIC_TYPE_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(self.normalize(alias))}\b", q) for alias in aliases):
                filters.append({"column": col, "operator": "contains", "value": code})
        return filters

    def _extract_month_filters(self, q: str, columns: list[str]) -> list[dict[str, Any]]:
        col = self.resolve_column(columns, ["mes"])
        if not col:
            return []
        filters: list[dict[str, Any]] = []
        # 12/2025, 12-2025, 2025-12
        m = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", q)
        if m:
            filters.append({"column": col, "operator": "eq", "value": f"{m.group(1)}-{m.group(2).zfill(2)}"})
        m = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", q)
        if m:
            filters.append({"column": col, "operator": "eq", "value": f"{m.group(2)}-{m.group(1).zfill(2)}"})
        for name, num in self.MONTHS.items():
            if re.search(rf"\b{name}\b", q):
                y = re.search(r"\b(20\d{2})\b", q)
                if y:
                    filters.append({"column": col, "operator": "eq", "value": f"{y.group(1)}-{num}"})
                else:
                    filters.append({"column": col, "operator": "contains", "value": f"-{num}"})
        return filters

    def _extract_type_filters(self, question: str, columns: list[str]) -> list[dict[str, Any]]:
        q = self.normalize(question)
        col = self.resolve_column(columns, ["tipo"])
        if not col:
            return []
        filters = []
        # Ex.: "tipo Normal", "do tipo Normal", "tipo = Normal".
        m = re.search(r"\b(?:do|da|de)?\s*tipo\s*(?:=|eh|e|é|:)?\s*([a-zA-ZÀ-ÿ0-9_\-/]+)", question or "", re.IGNORECASE)
        if m:
            value = m.group(1).strip()
            if self._valid_value(value):
                filters.append({"column": col, "operator": "eq", "value": value})
        # Normal/Standard/Emergencial etc quando aparece com CHG/change/mudança.
        if re.search(r"\bnormal\b", q) and any(x in q for x in ["chg", "change", "mudanca", "mudancas", "changes"]):
            filters.append({"column": col, "operator": "eq", "value": "Normal"})
        return filters

    def _extract_priority_filters(self, q: str, columns: list[str]) -> list[dict[str, Any]]:
        col = self.resolve_column(columns, ["prioridade"])
        if not col:
            return []
        m = re.search(r"\bP[0-9]\b", q.upper())
        if m:
            return [{"column": col, "operator": "contains", "value": m.group(0)}]
        return []

    def _detect_group_by(self, q: str, columns: list[str]) -> str | None:
        if " por " not in q and "agrup" not in q and "distribu" not in q:
            return None
        raw = None
        if " por " in q:
            raw = q.split(" por ", 1)[1]
            raw = re.split(r"[\?\.,;]", raw)[0]
        elif "categoria" in q:
            raw = "categoria"
        elif "status" in q:
            raw = "status"
        elif "tipo" in q:
            raw = "tipo"
        if not raw:
            return None
        return self.resolve_column(columns, [raw.strip(), raw.strip().split()[0]])

    def resolve_column(self, columns: list[str], candidates: Iterable[str]) -> str | None:
        if not columns:
            return None
        norm_columns = {col: self.normalize(col) for col in columns}
        terms: list[str] = []
        for candidate in candidates:
            if not candidate:
                continue
            c_norm = self.normalize(str(candidate))
            terms.append(c_norm)
            for canonical, aliases in self.FIELD_ALIASES.items():
                if c_norm == self.normalize(canonical) or c_norm in [self.normalize(a) for a in aliases]:
                    terms.append(self.normalize(canonical))
                    terms.extend([self.normalize(a) for a in aliases])
        best_col = None
        best_score = 0.0
        for col, norm_col in norm_columns.items():
            for term in terms:
                if not term:
                    continue
                score = SequenceMatcher(None, norm_col, term).ratio()
                if term == norm_col:
                    score += 1.0
                elif term in norm_col or norm_col in term:
                    score += 0.45
                if score > best_score:
                    best_col = col
                    best_score = score
        return best_col if best_score >= 0.58 else None

    def _valid_value(self, value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip().strip('"\'')
        if not text:
            return False
        norm = self.normalize(text)
        if norm in self.STOP_FILTER_VALUES:
            return False
        if len(norm) <= 1:
            return False
        return True

    def _is_valid_filter(self, filt: dict[str, Any]) -> bool:
        return bool(filt.get("column")) and self._valid_value(filt.get("value"))

    def _dedupe_filters(self, filters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen = set()
        for f in filters:
            if not self._is_valid_filter(f):
                continue
            key = (f.get("column"), f.get("operator", "contains"), self.normalize(str(f.get("value"))))
            if key in seen:
                continue
            seen.add(key)
            out.append({"column": f.get("column"), "operator": f.get("operator", "contains"), "value": f.get("value")})
        return out
