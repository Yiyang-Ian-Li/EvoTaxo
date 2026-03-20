from __future__ import annotations

from typing import Any, Dict, Optional

from .llm import LLMClient
from .utils import parse_json_object


def ask_json_with_retries(
    llm: LLMClient,
    prompt: str,
    system_prompt: str,
    max_parse_attempts: int,
    model_override: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    attempts = max(1, max_parse_attempts)
    for i in range(attempts):
        p = prompt
        if i > 0:
            p = prompt + "\n\nPrevious response was invalid JSON. Return one valid JSON object only."
        resp = llm.chat(
            p,
            response_format={"type": "json_object"},
            system_prompt=system_prompt,
            model_override=model_override,
        )
        if not resp:
            continue
        obj = parse_json_object(resp)
        if isinstance(obj, dict):
            return obj
    return None
