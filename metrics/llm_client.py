from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class EvalLLMConfig:
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    api_url: Optional[str] = None
    model: str = "gpt-4o-mini"
    timeout_s: int = 60
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    temperature: float = 0.0


class EvalLLMClient:
    def __init__(self, cfg: EvalLLMConfig):
        self.cfg = cfg
        self.provider = (cfg.provider or "openai").strip().lower()
        self.api_key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
        if not self.api_key and self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key and any(ord(ch) >= 128 for ch in self.api_key):
            self.api_key = None
        self.openai_client: Optional[OpenAI] = None
        if self.provider == "openai" and self.api_key:
            self.openai_client = OpenAI(api_key=self.api_key, base_url=self._openai_base_url())

    def _api_url(self) -> str:
        url = (self.cfg.api_url or "").strip()
        return url or DEFAULT_OPENAI_API_URL

    def _openai_base_url(self) -> str:
        url = self._api_url()
        suffix = "/chat/completions"
        return url[: -len(suffix)] if url.endswith(suffix) else url

    def available(self) -> bool:
        return self.provider == "openai" and bool(self.api_key)

    def chat(self, prompt: str) -> Optional[str]:
        if not self.available():
            return None
        return self._chat_openai(prompt)

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
