from __future__ import annotations

from typing import Any, Dict, List, Optional

from llm import LLMClient
from utils import parse_json_object


def ask_json_with_retries(
    llm: LLMClient,
    prompt: str,
    system_prompt: str,
    max_parse_attempts: int,
    trace: Optional[List[Dict[str, Any]]] = None,
    trace_meta: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    attempts = max(1, max_parse_attempts)
    for i in range(attempts):
        p = prompt
        if i > 0:
            p = prompt + "\n\nPrevious response was invalid JSON. Return one valid JSON object only."
        resp = llm.chat(p, response_format={"type": "json_object"}, system_prompt=system_prompt)
        if trace is not None and llm.cfg.trace_mode != "off":
            resp_len = len(resp or "")
            prompt_len = len(p)
            trace_row = {
                "kind": "llm_json_attempt",
                "trace_meta": trace_meta or {},
                "attempt": i + 1,
                "max_attempts": attempts,
                "prompt_len": prompt_len,
                "response_len": resp_len,
            }
            if llm.cfg.trace_mode == "full":
                trace_row["prompt"] = p
                trace_row["response_text"] = resp
            elif llm.cfg.trace_mode == "compact":
                max_chars = max(0, int(llm.cfg.trace_max_chars))
                trace_row["prompt_head"] = p[:max_chars]
                trace_row["response_head"] = (resp or "")[:max_chars]
            trace.append(trace_row)
        if not resp:
            continue
        obj = parse_json_object(resp)
        if isinstance(obj, dict):
            if trace is not None and llm.cfg.trace_mode != "off":
                trace.append(
                    {
                        "kind": "llm_json_success",
                        "trace_meta": trace_meta or {},
                        "attempt": i + 1,
                        "keys": sorted([str(k) for k in obj.keys()]),
                    }
                )
            return obj
    if trace is not None and llm.cfg.trace_mode != "off":
        trace.append(
            {
                "kind": "llm_json_failed",
                "trace_meta": trace_meta or {},
                "max_attempts": attempts,
            }
        )
    return None
