from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


class LLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.last_error: str | None = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.client),
            "provider": "openai" if self.client else "fallback",
            "model": self.model if self.client else None,
            "last_error": self.last_error,
        }

    def generate_chat(self, system_prompt: str, history: list[dict[str, str]], user_prompt: str, temperature: float = 0.2) -> str | None:
        if not self.client:
            return None
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-8:])
        messages.append({"role": "user", "content": user_prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=messages,
            )
            self.last_error = None
            return response.choices[0].message.content or ""
        except Exception as exc:
            self.last_error = str(exc)
            return None

    def generate_json(self, system_prompt: str, user_prompt: str, history: list[dict[str, str]] | None = None) -> dict[str, Any] | None:
        if not self.client:
            return None
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=messages,
            )
            self.last_error = None
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:
            self.last_error = str(exc)
            return None
