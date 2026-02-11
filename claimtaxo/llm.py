from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

from .config import LLMConfig


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.api_key = os.getenv("OPENAI_API_KEY")

    def available(self) -> bool:
        return self.cfg.enabled and bool(self.api_key)

    def chat(self, prompt: str, response_format: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if not self.available():
            return None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if response_format:
            payload["response_format"] = response_format
        try:
            resp = requests.post(
                self.cfg.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.cfg.timeout_s,
            )
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        return choices[0].get("message", {}).get("content", None)
