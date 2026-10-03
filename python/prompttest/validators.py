"""Deterministic format checks — no model call. Used for `format` assertions."""
from __future__ import annotations
import json

from .model import Assertion, Verdict
from .verdicts import Result


def check_format(a: Assertion, output: str) -> Result:
    spec = a.spec
    problems = []
    parsed = None

    if spec.get("json"):
        try:
            parsed = json.loads(output)
        except (ValueError, TypeError) as e:
            problems.append(f"invalid JSON ({e})")

    required = spec.get("required")
    if required:
        if parsed is None:
            try:
                parsed = json.loads(output)
            except (ValueError, TypeError):
                problems.append("required fields: output is not JSON")
        if isinstance(parsed, dict):
            missing = [k for k in required if k not in parsed]
            if missing:
                problems.append(f"missing fields: {missing}")

    max_len = spec.get("max_len")
    if max_len is not None and len(output) > max_len:
        problems.append(f"length {len(output)} > max_len {max_len}")

    min_len = spec.get("min_len")
    if min_len is not None and len(output) < min_len:
        problems.append(f"length {len(output)} < min_len {min_len}")

    v = Verdict.FAIL if problems else Verdict.PASS
    return Result(a.qid, a.label, v, "; ".join(problems) or "ok", 1.0)
