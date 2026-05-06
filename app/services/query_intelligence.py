from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

import pandas as pd


@dataclass
class PlannedQuery:
    use_tabular: bool
    intent: str
    filters: list[dict[str, Any]]
    group_by: str | None = None
    limit: int | None = None
    followup: bool = False
    confidence: float = 0.0
    reason: str = ""


class QueryIntelligenceService:
    """Camada genérica de inteligência para consulta tabular/RAG.

    Objetivo:
    - NÃO adicionar conhecimento externo ao agente.
    - Traduzir linguagem natural em filtros confiáveis com base nas colunas/valores da própria base.
    - Evitar filtros acidentais como "qual", "tipo", "me", "descreva".
    - Permitir follow-up contextual somente quando a pergunta realmente referencia o resultado anterior.
    """

    STOPWORDS = {
        "a", "ao", "aos", "as", "com", "como", "da", "das", "de", "dele", "deles", "dela", "delas",
        "do", "dos", "e", "em", "entre", "esse", "essa", "esses", "essas", "este", "esta", "estes", "estas",
        "eu", "me", "mim", "minha", "meu", "na", "nas", "no", "nos", "o", "os", "ou", "para", "por",
        "qual", "quais", "quando", "quanto", "quantos", "quantas", "que", "quem", "se", "sao", "são", "ser",
        "sobre", "contexto", "base", "bases", "conhecimento", "conhecimentos", "documento", "documentos",
        "registro", "registros", "linha", "linhas", "total", "quantidade", "numero", "número", "contar",
        "liste", "listar", "mostre", "descreva", "descrever", "detalhe", "detalhar", "explique", "fale",
        "cada", "um", "uma", "uns", "umas", "eles", "elas", "isso", "isto", "aquilo", "deste", "dessa", "desse",
        "tipo", "categoria", "status", "estado", "mes", "mês", "ano", "dia",
        "tem", "temos", "existe", "existem", "há", "ha", "foi", "foram", "seria", "seriam",
    }

    FOLLOWUP_MARKERS = {
        "eles", "elas", "deles", "delas", "destes", "destas", "desses", "dessas", "cada um", "cada uma",
        "cada um deles", "cada uma delas", "os mesmos", "as mesmas", "esses registros", "essas linhas",
        "me descreva", "descreva eles", "descreva elas", "detalhe eles", "detalhe elas", "detalhe cada",
        "liste eles", "liste elas", "listar eles", "listar elas", "desses", "dessas",
    }

    COUNT_MARKERS = {"quantos", "quantas", "quantidade", "total", "contar", "numero", "número", "qtd"}
    LIST_MARKERS = {"listar", "liste", "quais", "mostre", "descreva", "detalhe", "detalhar", "descrever", "fale"}
    GROUP_MARKERS = {"por", "agrup", "distribuicao", "distribuição", "ranking", "top"}
    BASE_CONTEXT_MARKERS = {"contexto dessa base", "contexto da base", "sobre o que se trata", "descreva a base", "descrever a base"}

    MONTHS = {
        "janeiro": "01", "jan": "01",
        "fevereiro": "02", "fev": "02",
        "marco": "03", "março": "03", "mar": "03",
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

    # Sinônimos gerais de domínio, sem vínculo com um agente específico.
    ENTITY_SYNONYMS = {
        "CHG": ["chg", "change", "changes", "mudanca", "mudancas", "mudança", "mudanças", "alteracao", "alterações", "alteracao"],
        "INC": ["inc", "incidente", "incidentes", "chamado", "chamados", "ocorrencia", "ocorrencias", "ocorrência", "ocorrências"],
        "REQ": ["req", "requisicao", "requisicoes", "requisição", "requisições", "solicitacao", "solicitações", "solicitação"],
    }

    CODE_PATTERN = re.compile(r"\b([A-Z]{2,10}\d{3,12}|[A-Z]{1,5}\d{2,8}|\d{4,}[A-Z]*)\b", re.IGNORECASE)

    def norm(self, value: Any) -> str:
        text = "" if value is None else str(value)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9]+", " ", text)
        return " ".join(text.split())

    def is_followup(self, question: str) -> bool:
        q = self.norm(question)
        return any(marker in q for marker in self.FOLLOWUP_MARKERS)

    def is_base_context_question(self, question: str) -> bool:
        q = self.norm(question)
        return any(marker in q for marker in self.BASE_CONTEXT_MARKERS)

    def infer_intent(self, question: str) -> str:
        q = self.norm(question)
        if self.is_base_context_question(question):
            return "describe_base"
        if any(marker in q for marker in self.COUNT_MARKERS):
            return "count"
        if (" por " in f" {q} " and any(marker in q for marker in self.COUNT_MARKERS | {"total"})) or any(m in q for m in ["distribuicao", "distribuição", "agrupe", "agrupado", "ranking"]):
            return "group"
        if any(marker in q for marker in self.LIST_MARKERS):
            return "list"
        if self.CODE_PATTERN.search(question):
            return "lookup"
        return "document"

    def has_followup_reference(self, question: str) -> bool:
        """Retorna True somente quando a pergunta depende claramente de um resultado anterior."""
        q = self.norm(question)
        strong_markers = [
            "cada uma delas", "cada um deles", "descreva eles", "descreva elas",
            "detalhe eles", "detalhe elas", "liste eles", "liste elas",
            "essas changes", "esses incidentes", "esses registros", "essas linhas",
            "delas", "deles", "destas", "destes", "dessas", "desses",
            "elas", "eles", "os mesmos", "as mesmas",
        ]
        return any(marker in q for marker in strong_markers)

    def is_independent_question(self, question: str) -> bool:
        """Perguntas independentes não devem herdar filtros anteriores."""
        q = self.norm(question)
        starters = (
            "qual ", "quais ", "quanto ", "quantos ", "quantas ",
            "me explique", "explique", "fale sobre", "o que ", "do que ",
            "contexto", "sobre o que", "me fale sobre", "liste o total",
        )
        if q.startswith(starters):
            return True
        if self.is_base_context_question(question):
            return True
        return False

    def should_use_tabular(self, question: str, columns: list[str]) -> bool:
        intent = self.infer_intent(question)
        if intent in {"count", "list", "group", "describe_base", "lookup"}:
            return True
        q = self.norm(question)
        joined_cols = " ".join(self.norm(c) for c in columns)
        return any(tok in joined_cols for tok in q.split() if len(tok) > 3 and tok not in self.STOPWORDS)

    def extract_codes(self, question: str) -> list[str]:
        out = []
        seen = set()
        for m in self.CODE_PATTERN.finditer(question or ""):
            code = m.group(1).upper().strip()
            if code and code not in seen and len(code) >= 3:
                seen.add(code)
                out.append(code)
        return out

    def extract_month(self, question: str) -> str | None:
        q = self.norm(question)
        # yyyy-mm direto
        m = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", q)
        if m:
            return f"{m.group(1)}-{int(m.group(2)):02d}"
        # mm/yyyy ou m/yyyy
        m = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d{2})\b", q)
        if m:
            return f"{m.group(2)}-{int(m.group(1)):02d}"
        # mês por extenso + ano
        for name, num in self.MONTHS.items():
            if re.search(rf"\b{name}\b", q):
                year = re.search(r"\b(20\d{2})\b", q)
                if year:
                    return f"{year.group(1)}-{num}"
                return num
        # "em 12 de 2025" / "12 2025"
        m = re.search(r"\b(0?[1-9]|1[0-2])\s+(?:de\s+)?(20\d{2})\b", q)
        if m:
            return f"{m.group(2)}-{int(m.group(1)):02d}"
        return None

    def column_score(self, column: str, candidates: Iterable[str]) -> float:
        col = self.norm(column)
        best = 0.0
        for cand in candidates:
            cn = self.norm(cand)
            if not cn:
                continue
            score = SequenceMatcher(None, col, cn).ratio()
            if cn in col or col in cn:
                score += 0.45
            best = max(best, score)
        return best

    def resolve_column(self, columns: list[str], candidates: Iterable[str], min_score: float = 0.55) -> str | None:
        best_col, best_score = None, 0.0
        for col in columns:
            score = self.column_score(col, candidates)
            if score > best_score:
                best_col, best_score = col, score
        return best_col if best_score >= min_score else None

    def preferred_column_for_entity(self, columns: list[str]) -> str | None:
        return self.resolve_column(columns, ["codigo_tipo", "tipo_codigo", "code_type", "tipo do codigo", "classe_codigo"], 0.52)

    def preferred_code_columns(self, columns: list[str]) -> list[str]:
        prefs = []
        for candidates in [
            ["numero", "número", "codigo_principal", "codigo", "id", "identificador", "chave"],
            ["codigo_principal", "codigo", "transacao_z", "identificadores", "article_text", "conteudo", "texto"],
        ]:
            col = self.resolve_column(columns, candidates, 0.52)
            if col and col not in prefs:
                prefs.append(col)
        return prefs

    def value_exists(self, df: pd.DataFrame, column: str, value: str, operator: str = "contains") -> bool:
        if column not in df.columns:
            return False
        series = df[column].astype(str).fillna("")
        if operator == "eq":
            return bool((series.str.lower() == str(value).lower()).any())
        return bool(series.str.contains(re.escape(str(value)), case=False, na=False, regex=True).any())

    def detect_entity_filter(self, question: str, df: pd.DataFrame) -> dict[str, Any] | None:
        columns = list(df.columns)
        q = self.norm(question)
        entity_col = self.preferred_column_for_entity(columns)
        if not entity_col:
            return None
        for canonical, terms in self.ENTITY_SYNONYMS.items():
            if any(re.search(rf"\b{re.escape(self.norm(term))}\b", q) for term in terms):
                if self.value_exists(df, entity_col, canonical, operator="contains"):
                    return {"column": entity_col, "operator": "contains", "value": canonical, "confidence": 0.95, "source": "entity_synonym"}
        # códigos explícitos também podem indicar o tipo pelo prefixo se existir na própria tabela.
        codes = self.extract_codes(question)
        for code in codes:
            prefix = re.match(r"^[A-Z]+", code)
            if prefix:
                val = prefix.group(0)
                if self.value_exists(df, entity_col, val, operator="contains"):
                    return {"column": entity_col, "operator": "contains", "value": val, "confidence": 0.86, "source": "code_prefix"}
        return None

    def detect_type_value_filter(self, question: str, df: pd.DataFrame) -> dict[str, Any] | None:
        columns = list(df.columns)
        q = self.norm(question)
        # Só tenta quando o usuário fala explicitamente "tipo X", "do tipo X", "tipo = X".
        m = re.search(r"\b(?:do\s+)?tipo\s*(?:=|eh|e|é|de)?\s+([a-zA-Z0-9_\-\/]+)\b", q)
        if not m:
            return None
        value = m.group(1).strip()
        if not self.is_valid_filter_value(value):
            return None
        # Não usar categoria para "tipo X"; preferir coluna chamada tipo/status/estado, mas não codigo_tipo.
        candidates = ["tipo", "tipo_change", "tipo_registro", "classe", "natureza"]
        possible = [c for c in columns if self.norm(c) not in {"codigo tipo", "codigo_tipo"}]
        col = self.resolve_column(possible, candidates, 0.58)
        if col and self.value_exists(df, col, value, operator="contains"):
            return {"column": col, "operator": "contains", "value": value, "confidence": 0.9, "source": "explicit_type"}
        return None

    def detect_month_filter(self, question: str, df: pd.DataFrame) -> dict[str, Any] | None:
        month = self.extract_month(question)
        if not month:
            return None
        columns = list(df.columns)
        col = self.resolve_column(columns, ["mes", "mês", "competencia", "competência", "periodo", "período", "data", "data_inicio", "created_on"], 0.52)
        if not col:
            return None
        operator = "eq" if re.fullmatch(r"20\d{2}-\d{2}", month) else "contains"
        return {"column": col, "operator": operator, "value": month, "confidence": 0.9, "source": "month"}

    def detect_code_filters(self, question: str, df: pd.DataFrame) -> list[dict[str, Any]]:
        columns = list(df.columns)
        out = []
        for code in self.extract_codes(question):
            for col in self.preferred_code_columns(columns):
                if self.value_exists(df, col, code, operator="contains"):
                    out.append({"column": col, "operator": "contains", "value": code, "confidence": 0.95, "source": "explicit_code"})
                    break
        return out

    def detect_group_by(self, question: str, columns: list[str]) -> str | None:
        q = self.norm(question)
        if " por " not in f" {q} " and not any(x in q for x in ["agrupe", "agrupado", "distribuicao", "distribuição", "ranking"]):
            return None
        raw = None
        if " por " in f" {q} ":
            raw = q.split(" por ", 1)[1]
            raw = re.split(r"[\?\.,;]", raw)[0].strip()
        elif "categoria" in q:
            raw = "categoria"
        elif "status" in q:
            raw = "status"
        elif "estado" in q:
            raw = "estado"
        elif "tipo" in q:
            raw = "tipo"
        if not raw:
            return None
        # Remover palavras inúteis após o "por".
        tokens = [t for t in raw.split() if t not in self.STOPWORDS]
        candidate = " ".join(tokens[:3]) if tokens else raw
        return self.resolve_column(columns, [candidate, raw], 0.48)

    def is_valid_filter_value(self, value: Any) -> bool:
        if value is None:
            return False
        text = str(value).strip()
        if not text:
            return False
        n = self.norm(text)
        if n in self.STOPWORDS:
            return False
        if len(n) <= 2 and not re.fullmatch(r"p\d+", n):
            return False
        return True

    def sanitize_filters(self, filters: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen = set()
        for f in filters or []:
            col = f.get("column")
            value = f.get("value")
            op = f.get("operator") or "contains"
            conf = float(f.get("confidence") or 0.5)
            if col not in df.columns:
                continue
            if not self.is_valid_filter_value(value):
                continue
            # Evita filtros acidentais em categoria quando o valor veio de palavra da pergunta.
            if self.norm(col) in {"categoria", "tema", "assunto"} and self.norm(value) in self.STOPWORDS:
                continue
            key = (col, op, str(value).lower())
            if key in seen:
                continue
            seen.add(key)
            out.append({"column": col, "operator": op, "value": value, "confidence": conf, "source": f.get("source")})
        return out

    def build_plan(self, question: str, df: pd.DataFrame, last_context: dict[str, Any] | None = None) -> PlannedQuery:
        columns = list(df.columns)
        intent = self.infer_intent(question)

        # Follow-up só é verdadeiro quando a pergunta faz referência clara ao resultado anterior.
        # Perguntas independentes, mesmo depois de uma consulta tabular, não herdam filtros.
        followup = bool(last_context) and self.has_followup_reference(question) and not self.is_independent_question(question)

        filters: list[dict[str, Any]] = []
        reason_parts: list[str] = []

        if intent == "document" and not self.should_use_tabular(question, columns):
            return PlannedQuery(False, "document", [], confidence=0.2, reason="not_tabular")

        # Pergunta sobre o contexto/base: sempre consulta ampla, sem herdar filtros.
        if intent == "describe_base":
            return PlannedQuery(True, "describe_base", [], limit=150, confidence=0.9, reason="base_context")

        code_filters = self.detect_code_filters(question, df)
        if code_filters:
            filters.extend(code_filters)
            reason_parts.append("explicit_code")
            followup = False

        entity_filter = self.detect_entity_filter(question, df)
        if entity_filter:
            filters.append(entity_filter)
            reason_parts.append("entity")

        month_filter = self.detect_month_filter(question, df)
        if month_filter:
            filters.append(month_filter)
            reason_parts.append("month")

        type_filter = self.detect_type_value_filter(question, df)
        if type_filter:
            filters.append(type_filter)
            reason_parts.append("type")

        # Se a pergunta trouxe filtros novos explícitos, ela é uma consulta nova, não continuação.
        has_new_structured_filter = bool(code_filters or entity_filter or month_filter or type_filter)
        if has_new_structured_filter and not self.has_followup_reference(question):
            followup = False

        # Follow-up: reaproveita apenas filtros estruturais e confiáveis do último resultado útil.
        # Nunca herda filtros acidentais, filtros que zeraram resultado, nem contexto se a entidade mudou.
        if followup and last_context and last_context.get("filters"):
            previous = last_context.get("filters") or []
            current_entity_values = {
                str(f.get("value")).upper()
                for f in filters
                if f.get("source") in {"entity_synonym", "code_prefix"}
            }
            for f in previous:
                source = f.get("source")
                value = str(f.get("value") or "")
                col = f.get("column")
                if not col or col not in df.columns:
                    continue
                if current_entity_values and str(value).upper() in {"INC", "CHG", "REQ"} and str(value).upper() not in current_entity_values:
                    continue
                if source in {"entity_synonym", "month", "explicit_type", "explicit_code", "code_prefix"} or float(f.get("confidence") or 0) >= 0.80:
                    filters.append(dict(f))
            reason_parts.append("followup")

        filters = self.sanitize_filters(filters, df)

        group_by = self.detect_group_by(question, columns)
        if group_by:
            intent = "group"
        elif intent == "lookup":
            intent = "list"
        elif intent == "document" and filters:
            intent = "list"
        elif intent == "document":
            return PlannedQuery(False, "document", [], confidence=0.2, reason="document_fallback")

        limit = 20
        if intent in {"list", "describe_base"}:
            limit = 150
        return PlannedQuery(
            True,
            intent,
            filters,
            group_by=group_by,
            limit=limit,
            followup=followup,
            confidence=0.88 if filters or intent in {"describe_base", "group"} else 0.65,
            reason="+".join(reason_parts) or "heuristic",
        )
