from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from openai import OpenAI


DEFAULT_CUSTOM_API_URL = "https://openwebui.crc.nd.edu/api/v1/chat/completions"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class EvalLLMConfig:
    provider: str = "custom"
    api_key_env: str = "OPENAI_API_KEY"
    api_url: str = DEFAULT_CUSTOM_API_URL
    model: str = "gpt-oss:120b"
    timeout_s: int = 60
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    temperature: float = 0.0


class EvalLLMClient:
    def __init__(self, cfg: EvalLLMConfig):
        self.cfg = cfg
        self.provider = (cfg.provider or "custom").strip().lower()
        self.api_key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
        if not self.api_key:
            if self.provider == "custom":
                self.api_key = os.getenv("OPENWEBUI_API_KEY") or os.getenv("OPENAI_API_KEY")
            else:
                self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and any(ord(ch) >= 128 for ch in self.api_key):
            self.api_key = None
        self.openai_client: Optional[OpenAI] = None
        if self.provider == "openai" and self.api_key:
            self.openai_client = OpenAI(api_key=self.api_key, base_url=self._openai_base_url())

    def _api_url(self) -> str:
        url = (self.cfg.api_url or "").strip()
        if url:
            if self.provider == "openai" and url == DEFAULT_CUSTOM_API_URL:
                return DEFAULT_OPENAI_API_URL
            return url
        return DEFAULT_OPENAI_API_URL if self.provider == "openai" else DEFAULT_CUSTOM_API_URL

    def _openai_base_url(self) -> str:
        url = self._api_url()
        suffix = "/chat/completions"
        return url[: -len(suffix)] if url.endswith(suffix) else url

    def available(self) -> bool:
        return self.provider in {"custom", "openai"} and bool(self.api_key)

    def chat(self, prompt: str) -> Optional[str]:
        if not self.available():
            return None
        if self.provider == "openai":
            return self._chat_openai(prompt)
        return self._chat_custom(prompt)

    def _chat_custom(self, prompt: str) -> Optional[str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
        }
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(
                    self._api_url(),
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.cfg.timeout_s,
                )
            except requests.RequestException:
                resp = None
            if resp is not None and resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", None)
            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_backoff_s)
        return None

    def _chat_openai(self, prompt: str) -> Optional[str]:
        if self.openai_client is None:
            return None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self.openai_client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.cfg.temperature,
                    timeout=self.cfg.timeout_s,
                )
                if resp.choices:
                    return resp.choices[0].message.content
            except Exception:
                pass
            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_backoff_s)
        return None
