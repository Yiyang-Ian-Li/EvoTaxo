from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, Optional

import requests
from openai import OpenAI

from config import LLMConfig

DEFAULT_CUSTOM_API_URL = "https://openwebui.crc.nd.edu/api/v1/chat/completions"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.logger = logging.getLogger("claimtaxo_v2")
        self.provider = (cfg.provider or "custom").strip().lower()
        self.openai_client: Optional[OpenAI] = None
        self.api_key = os.getenv(cfg.api_key_env) if cfg.api_key_env else None
        if not self.api_key:
            if self.provider == "custom":
                self.api_key = os.getenv("OPENWEBUI_API_KEY") or os.getenv("OPENAI_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == "openrouter":
                self.api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        if self.api_key and any(ord(ch) >= 128 for ch in self.api_key):
            # Requests/http headers require latin-1 encodable values.
            self.api_key = None
        if self.provider in {"openai", "openrouter"} and self.api_key:
            base_url = self._openai_base_url()
            self.openai_client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _api_url(self) -> str:
        url = (self.cfg.api_url or "").strip()
        if url:
            if self.provider == "openai" and url == DEFAULT_CUSTOM_API_URL:
                return DEFAULT_OPENAI_API_URL
            if self.provider == "openrouter" and url == DEFAULT_CUSTOM_API_URL:
                return DEFAULT_OPENROUTER_API_URL
            return url
        if self.provider == "openai":
            return DEFAULT_OPENAI_API_URL
        if self.provider == "openrouter":
            return DEFAULT_OPENROUTER_API_URL
        return DEFAULT_CUSTOM_API_URL

    def _openai_base_url(self) -> str:
        url = self._api_url()
        if url.endswith("/chat/completions"):
            return url[: -len("/chat/completions")]
        return url

    def available(self) -> bool:
        return self.cfg.enabled and self.provider in {"custom", "openai", "openrouter"} and bool(self.api_key)

    def chat(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        if not self.available():
            return None
        if self.provider in {"openai", "openrouter"}:
            return self._chat_openai(prompt, response_format, system_prompt, model_override=model_override)
        return self._chat_custom(prompt, response_format, system_prompt, model_override=model_override)

    def _chat_custom(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": (model_override or self.cfg.model),
            "messages": messages,
            "temperature": self.cfg.temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = requests.post(
                    self._api_url(),
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.cfg.timeout_s,
                )
            except requests.RequestException:
                self.logger.warning(
                    "LLM custom request failed attempt=%d/%d",
                    attempt + 1,
                    self.cfg.max_retries + 1,
                )
                resp = None

            if resp is not None and resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", None)
            elif resp is not None:
                self.logger.warning(
                    "LLM custom non-200 status=%s attempt=%d/%d",
                    resp.status_code,
                    attempt + 1,
                    self.cfg.max_retries + 1,
                )

            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_backoff_s)

        return None

    def _chat_openai(
        self,
        prompt: str,
        response_format: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        model_override: Optional[str] = None,
    ) -> Optional[str]:
        if self.openai_client is None:
            return None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs: Dict[str, Any] = {
            "model": (model_override or self.cfg.model),
            "messages": messages,
            "temperature": self.cfg.temperature,
            "timeout": self.cfg.timeout_s,
        }
        if response_format:
            kwargs["response_format"] = response_format

        for attempt in range(self.cfg.max_retries + 1):
            try:
                resp = self.openai_client.chat.completions.create(**kwargs)
                if resp.choices:
                    return resp.choices[0].message.content
            except Exception:
                self.logger.warning(
                    "LLM openai request failed attempt=%d/%d",
                    attempt + 1,
                    self.cfg.max_retries + 1,
                )

            if attempt < self.cfg.max_retries:
                time.sleep(self.cfg.retry_backoff_s)

        return None
