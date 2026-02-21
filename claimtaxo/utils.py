from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class JsonlSink:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").close()
        self.count = 0

    def append(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.count += 1


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    txt = raw.strip()
    if not txt:
        return None

    fenced = re.sub(r"^```(?:json)?\\s*", "", txt, flags=re.IGNORECASE)
    fenced = re.sub(r"\\s*```$", "", fenced).strip()

    for candidate in (txt, fenced):
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            pass

    start = fenced.find("{")
    end = fenced.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(fenced[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None
