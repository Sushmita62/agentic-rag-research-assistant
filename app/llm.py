"""Thin Groq client via OpenAI-compatible endpoint. Swappable to OpenAI/local later
by changing base_url + api_key.
"""
from __future__ import annotations

from app.config import settings


class LLMClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        from openai import OpenAI

        key = api_key or settings.groq_api_key
        if not key:
            raise RuntimeError("GROQ_API_KEY not set. Put it in .env")
        self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        self.model = model or settings.groq_model

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        kwargs: dict = {"model": self.model, "messages": messages, "temperature": temperature}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
