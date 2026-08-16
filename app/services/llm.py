from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from openai import OpenAI


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class LLMCallResult:
    ok: bool
    content: str | None = None
    json_data: JsonDict | None = None
    error: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    attempts: int = 0


class LLMService:
    """
    LLM Gateway para o Gabbi/Nexus.

    Objetivo arquitetural:
    - OpenAI interpreta linguagem, intenção e contexto.
    - DuckDB/engine local executa cálculo e decisão determinística.
    - O LLM nunca deve ser a fonte da verdade numérica.

    Compatível com a classe antiga:
    - status()
    - generate_chat(...)
    - generate_json(...)

    Novos recursos:
    - generate_schema(...): Structured Outputs com JSON Schema + fallback seguro.
    - generate_analytic_plan(...): planner analítico genérico.
    - resolve_semantic_context(...): resolve follow-up/contexto conversacional.
    - generate_narrative(...): formata resultado estruturado sem recalcular.
    """

    DEFAULT_ANALYTIC_PLAN_SCHEMA: JsonDict = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "answerable",
            "intent",
            "metric",
            "entity",
            "filters",
            "group_by",
            "sort",
            "limit",
            "needs_context",
            "confidence",
            "reason",
        ],
        "properties": {
            "answerable": {"type": "boolean"},
            "intent": {
                "type": "string",
                "enum": [
                    "count",
                    "list",
                    "detail",
                    "rank",
                    "group",
                    "compare",
                    "summary",
                    "trend",
                    "correlation",
                    "forecast",
                    "causal_hypothesis",
                    "criteria_explanation",
                    "unknown",
                ],
            },
            "metric": {
                "type": "string",
                "enum": [
                    "count",
                    "distinct_count",
                    "impact_total",
                    "impact_max",
                    "mttr",
                    "parada_sistemica",
                    "change_related",
                    "priority_distribution",
                    "risk_score",
                    "status_distribution",
                    "unknown",
                ],
            },
            "entity": {
                "type": "string",
                "description": "Entidade principal consultada. Ex.: incidentes, changes, funcionalidade, grupo, causa, prioridade.",
            },
            "filters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["field", "operator", "value"],
                    "properties": {
                        "field": {"type": "string"},
                        "operator": {
                            "type": "string",
                            "enum": ["eq", "contains", "in", "gte", "lte", "between", "is_true", "is_false"],
                        },
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "array", "items": {"type": "string"}},
                            ]
                        },
                    },
                },
            },
            "group_by": {"type": "array", "items": {"type": "string"}},
            "sort": {
                "type": "object",
                "additionalProperties": False,
                "required": ["field", "direction"],
                "properties": {
                    "field": {"type": "string"},
                    "direction": {"type": "string", "enum": ["asc", "desc", "none"]},
                },
            },
            "limit": {"type": "integer", "minimum": 0, "maximum": 200},
            "needs_context": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
    }

    DEFAULT_CONTEXT_RESOLUTION_SCHEMA: JsonDict = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "resolved_question",
            "uses_previous_context",
            "inherited_fields",
            "confidence",
            "reason",
        ],
        "properties": {
            "resolved_question": {"type": "string"},
            "uses_previous_context": {"type": "boolean"},
            "inherited_fields": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
        },
    }

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.planner_model = os.getenv("OPENAI_PLANNER_MODEL", self.model).strip()
        self.narrative_model = os.getenv("OPENAI_NARRATIVE_MODEL", self.model).strip()
        self.timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
        self.store = os.getenv("OPENAI_STORE", "false").strip().lower() in {"1", "true", "yes", "sim", "on"}
        self.client = OpenAI(api_key=self.api_key, timeout=self.timeout) if self.api_key else None
        self.last_error: str | None = None
        self.last_latency_ms: int | None = None
        self.last_usage: JsonDict | None = None

    # ------------------------------------------------------------------
    # Compat/status
    # ------------------------------------------------------------------
    def status(self) -> JsonDict:
        return {
            "enabled": bool(self.client),
            "provider": "openai" if self.client else "fallback",
            "model": self.model if self.client else None,
            "planner_model": self.planner_model if self.client else None,
            "narrative_model": self.narrative_model if self.client else None,
            "store": self.store,
            "last_error": self.last_error,
            "last_latency_ms": self.last_latency_ms,
            "last_usage": self.last_usage,
        }

    def generate_chat(
        self,
        system_prompt: str,
        history: list[dict[str, str]],
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str | None:
        result = self._chat_text(
            model=self.narrative_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history,
            temperature=temperature,
        )
        return result.content if result.ok else None

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
    ) -> JsonDict | None:
        # Mantém compatibilidade, mas tenta schema livre com JSON mode.
        result = self._chat_json_object(
            model=self.planner_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history or [],
            temperature=0,
        )
        return result.json_data if result.ok else None

    # ------------------------------------------------------------------
    # Novos métodos de gateway
    # ------------------------------------------------------------------
    def generate_schema(
        self,
        *,
        schema_name: str,
        schema: JsonDict,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None = None,
        model: str | None = None,
        temperature: float = 0,
        fallback_to_json_object: bool = True,
    ) -> JsonDict | None:
        """Gera JSON aderente a JSON Schema com Structured Outputs.

        Se o modelo/SDK não aceitar json_schema, cai para json_object.
        """
        result = self._chat_json_schema(
            model=model or self.planner_model,
            schema_name=schema_name,
            schema=schema,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            history=history or [],
            temperature=temperature,
        )
        if result.ok:
            return result.json_data

        if fallback_to_json_object:
            fallback_prompt = (
                system_prompt.rstrip()
                + "\n\nResponda SOMENTE em JSON válido aderente a este JSON Schema: "
                + json.dumps(schema, ensure_ascii=False)
            )
            result = self._chat_json_object(
                model=model or self.planner_model,
                system_prompt=fallback_prompt,
                user_prompt=user_prompt,
                history=history or [],
                temperature=0,
            )
            return result.json_data if result.ok else None
        return None

    def generate_analytic_plan(
        self,
        *,
        question: str,
        schema_context: JsonDict,
        conversation_state: JsonDict | None = None,
        examples: list[JsonDict] | None = None,
    ) -> JsonDict | None:
        """Transforma pergunta livre em plano analítico genérico.

        A resposta é um plano; NÃO é o resultado final.
        O executor local deve validar colunas/filtros e executar em DuckDB.
        """
        system_prompt = """
Você é o planejador analítico do Gabbi/Nexus.
Sua função é interpretar a pergunta do usuário e gerar um plano JSON para uma engine determinística.
NUNCA calcule números finais. NUNCA invente campos fora do schema.
Use o estado conversacional apenas para resolver follow-ups, como "deles", "ele", "em setembro", "e no mês anterior".
Se a pergunta for aberta demais ou não for analítica, marque answerable=false e intent=unknown.
Campos como "dor", "sofreu", "deu problema" geralmente indicam ranking por impacto ou volume, conforme o contexto.
""".strip()
        payload = {
            "question": question,
            "schema_context": schema_context,
            "conversation_state": conversation_state or {},
            "examples": examples or [],
        }
        return self.generate_schema(
            schema_name="analytic_query_plan",
            schema=self.DEFAULT_ANALYTIC_PLAN_SCHEMA,
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            model=self.planner_model,
            temperature=0,
        )

    def resolve_semantic_context(
        self,
        *,
        question: str,
        conversation_state: JsonDict,
        history: list[dict[str, str]] | None = None,
    ) -> JsonDict | None:
        """Resolve perguntas curtas/ambíguas usando estado estruturado local."""
        system_prompt = """
Você é o resolvedor de contexto conversacional do Gabbi/Nexus.
Reescreva a pergunta atual de forma completa usando apenas o estado conversacional fornecido.
Não invente dados. Se não houver contexto suficiente, mantenha a pergunta original e confidence baixo.
""".strip()
        payload = {
            "question": question,
            "conversation_state": conversation_state or {},
        }
        return self.generate_schema(
            schema_name="context_resolution",
            schema=self.DEFAULT_CONTEXT_RESOLUTION_SCHEMA,
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            history=history or [],
            model=self.planner_model,
            temperature=0,
        )

    def generate_narrative(
        self,
        *,
        question: str,
        structured_result: JsonDict,
        conversation_state: JsonDict | None = None,
        mode: str = "executive",
    ) -> str | None:
        """Formata resultado determinístico em linguagem natural.

        Regra central: o LLM não pode alterar números, códigos ou listas.
        """
        system_prompt = """
Você é um analista sênior do Gabbi/Nexus.
Formate o resultado estruturado em português do Brasil.
REGRAS:
1. Não altere números, códigos, datas ou listas presentes no structured_result.
2. Se houver SQL/debug/technical, use apenas para explicar critério, não exponha tudo salvo se solicitado.
3. Se confidence for baixo, sinalize limitação.
4. Seja objetivo em modo executive.
""".strip()
        if mode == "technical":
            system_prompt += "\nInclua critério, filtros e observações técnicas relevantes."
        elif mode == "executive":
            system_prompt += "\nPriorize resposta curta, executiva e orientada à decisão."

        payload = {
            "question": question,
            "structured_result": structured_result,
            "conversation_state": conversation_state or {},
            "mode": mode,
        }
        result = self._chat_text(
            model=self.narrative_model,
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            history=[],
            temperature=0.1,
        )
        return result.content if result.ok else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _messages(self, system_prompt: str, user_prompt: str, history: list[dict[str, str]] | None, max_history: int) -> list[JsonDict]:
        messages: list[JsonDict] = [{"role": "system", "content": system_prompt or ""}]
        for item in (history or [])[-max_history:]:
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant", "system"} and content:
                # Evita system injection via histórico; system histórico vira assistant context.
                if role == "system":
                    role = "assistant"
                messages.append({"role": role, "content": str(content)})
        messages.append({"role": "user", "content": user_prompt or ""})
        return messages

    def _chat_text(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None,
        temperature: float,
    ) -> LLMCallResult:
        if not self.client:
            return LLMCallResult(ok=False, error="openai_client_disabled")
        return self._with_retries(
            lambda: self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=self._messages(system_prompt, user_prompt, history, max_history=6),
                **self._storage_kwargs(),
            ),
            parser=lambda response: response.choices[0].message.content or "",
            model=model,
            json_expected=False,
        )

    def _chat_json_object(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None,
        temperature: float,
    ) -> LLMCallResult:
        if not self.client:
            return LLMCallResult(ok=False, error="openai_client_disabled")
        return self._with_retries(
            lambda: self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=self._messages(system_prompt, user_prompt, history, max_history=4),
                **self._storage_kwargs(),
            ),
            parser=self._parse_json_response,
            model=model,
            json_expected=True,
        )

    def _chat_json_schema(
        self,
        *,
        model: str,
        schema_name: str,
        schema: JsonDict,
        system_prompt: str,
        user_prompt: str,
        history: list[dict[str, str]] | None,
        temperature: float,
    ) -> LLMCallResult:
        if not self.client:
            return LLMCallResult(ok=False, error="openai_client_disabled")

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": self._safe_schema_name(schema_name),
                "strict": True,
                "schema": schema,
            },
        }
        return self._with_retries(
            lambda: self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format=response_format,
                messages=self._messages(system_prompt, user_prompt, history, max_history=4),
                **self._storage_kwargs(),
            ),
            parser=self._parse_json_response,
            model=model,
            json_expected=True,
        )

    def _with_retries(self, call, parser, model: str, json_expected: bool) -> LLMCallResult:
        started = time.time()
        last_error = None
        attempts = max(1, self.max_retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                response = call()
                latency_ms = int((time.time() - started) * 1000)
                self.last_latency_ms = latency_ms
                self.last_usage = self._usage_dict(response)
                parsed = parser(response)
                self.last_error = None
                if json_expected:
                    return LLMCallResult(ok=True, json_data=parsed, model=model, latency_ms=latency_ms, attempts=attempt)
                return LLMCallResult(ok=True, content=parsed, model=model, latency_ms=latency_ms, attempts=attempt)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                self.last_error = last_error
                if attempt < attempts:
                    time.sleep(min(2.0, 0.25 * attempt + random.random() * 0.25))
        return LLMCallResult(ok=False, error=last_error, model=model, latency_ms=int((time.time() - started) * 1000), attempts=attempts)

    @staticmethod
    def _parse_json_response(response) -> JsonDict:
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return data if isinstance(data, dict) else {"value": data}

    @staticmethod
    def _usage_dict(response) -> JsonDict | None:
        usage = getattr(response, "usage", None)
        if not usage:
            return None
        try:
            return {
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "total_tokens": getattr(usage, "total_tokens", None),
            }
        except Exception:
            return None

    def _storage_kwargs(self) -> JsonDict:
        # Em muitos cenários enterprise/LGPD, prefira OPENAI_STORE=false.
        # Caso o SDK/modelo não aceite, _with_retries captura e o erro fica visível.
        return {"store": self.store}

    @staticmethod
    def _safe_schema_name(value: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in (value or "schema"))
        cleaned = cleaned.strip("_-") or "schema"
        return cleaned[:64]
