from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentRouteDecision:
    route: str
    reason: str
    confidence: float = 0.0


class AgentIntentRouter:
    """Router leve e genérico.

    Pode ser usado fora do KnowledgeStructuredStore quando o fluxo principal do chat
    precisar escolher entre RAG, SQL ou híbrido antes de chamar uma engine específica.
    """

    def classify(self, question: str, schema_fields: list[str] | None = None, agent_profile: dict[str, Any] | None = None) -> AgentRouteDecision:
        q = (question or "").lower()
        schema_text = " ".join(schema_fields or []).lower()
        analytic_markers = ["quant", "total", "qtd", "listar", "liste", "quais", "ranking", "top", "maior", "menor", "por "]
        semantic_markers = ["explique", "como funciona", "o que é", "resuma", "fale sobre", "manual", "procedimento", "regra"]
        analytic = any(m in q for m in analytic_markers) or any(f and f in q for f in schema_text.split())
        semantic = any(m in q for m in semantic_markers)
        if analytic and semantic:
            return AgentRouteDecision("hybrid", "analytic_and_semantic_markers", 0.85)
        if analytic:
            return AgentRouteDecision("analytic", "analytic_markers_or_schema_fields", 0.8)
        return AgentRouteDecision("semantic", "default_rag_semantic", 0.75)
