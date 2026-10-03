"""The assertion DSL: ergonomic builders + a YAML loader.

Every assertion compiles to a binary `noul` or a small `choice`/`score` (<=4
options). This keeps the judge in the zone where Laya is accurate.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pathlib
import yaml

from .model import Assertion
from .judge.base import JudgeQuestion


class DSLError(ValueError):
    pass


def _noul(qid, behavior, want_true, label):
    return Assertion(qid=qid, kind="noul", label=label,
                     instructions=f"Does the assistant output {behavior}?",
                     criteria={"true": f"it {behavior}", "false": "it does not"},
                     want_true=want_true)


def satisfies(behavior: str, qid: str | None = None) -> Assertion:
    return _noul(qid or _slug(behavior), behavior, True, f"satisfies: {behavior}")


def not_(behavior: str, qid: str | None = None) -> Assertion:
    return _noul(qid or "not_" + _slug(behavior), behavior, False, f"not: {behavior}")


def choice(question: str, among: list, want, qid: str | None = None) -> Assertion:
    if len(among) > 4:
        raise DSLError(f"choice has {len(among)} options; Laya is weak past ~4. "
                       "Split into binary satisfies/not_ assertions (one-vs-rest).")
    return Assertion(qid=qid or _slug(question), kind="choice",
                     label=f"choice: {question} == {want}",
                     instructions=question, criteria={o: str(o) for o in among}, want=want)


def score(question: str, rubric: list, min_level, qid: str | None = None) -> Assertion:
    if min_level not in rubric:
        raise DSLError(f"min_level {min_level!r} not in rubric {rubric}")
    return Assertion(qid=qid or _slug(question), kind="score",
                     label=f"score: {question} >= {min_level}",
                     instructions=question, criteria=list(rubric),
                     floor=rubric.index(min_level), rubric=list(rubric))


def format_(qid: str = "format", **spec) -> Assertion:
    """Deterministic check (no judge). spec: json=True, required=[...], max_len=N."""
    return Assertion(qid=qid, kind="format", label=f"format: {spec}", spec=spec)


def any_of(questions: list[str], *, threshold: float = 0.5, combine: str = "max",
           want_present: bool = True, qid: str | None = None,
           label: str | None = None) -> Assertion:
    """Ensemble detector: ask several binary questions (verbatim) and combine
    their P(true) (default MAX). The behavior is 'present' when the combined
    score >= `threshold`. PASS/FAIL is relative to `want_present`.

    Targeted sub-questions + MAX catch more attack variants than one broad
    question, and the threshold is a per-task dial you tune on held-out data.
    """
    if not questions:
        raise DSLError("any_of needs at least one question")
    return Assertion(qid=qid or _slug(questions[0]), kind="ensemble",
                     label=label or ("any_of" if want_present else "none_of"),
                     subs=list(questions), combine=combine, threshold=threshold,
                     want_true=want_present)


def none_of(questions: list[str], *, threshold: float = 0.5, combine: str = "max",
            qid: str | None = None, label: str | None = None) -> Assertion:
    """Guardrail form of `any_of`: PASS only when none of the questions fire."""
    return any_of(questions, threshold=threshold, combine=combine, want_present=False,
                  qid=qid, label=label)


def to_question(a: Assertion) -> JudgeQuestion:
    if a.kind == "format":
        raise DSLError("format assertions are checked deterministically, not judged")
    return JudgeQuestion(a.qid, a.kind, a.instructions, a.criteria)


# ---- YAML suites -----------------------------------------------------------

@dataclass
class Case:
    id: str
    input: Any
    output: str
    assertions: list[Assertion] = field(default_factory=list)


@dataclass
class Suite:
    name: str
    cases: list[Case] = field(default_factory=list)
    judge: dict = field(default_factory=dict)


_BUILDERS = {"satisfies": satisfies, "not": not_}


def _parse_assert(item: dict) -> Assertion:
    (key, val), = item.items() if len(item) == 1 else [(None, None)]
    if key in ("satisfies", "not"):
        return _BUILDERS[key](val)
    if key == "choice":
        return choice(val["question"], val["among"], val["want"])
    if key == "score":
        return score(val["question"], val["rubric"], val["min"])
    if key == "format":
        return format_(**val)
    if key in ("any_of", "none_of"):
        qs = val["questions"] if isinstance(val, dict) else val
        kw = {k: v for k, v in (val.items() if isinstance(val, dict) else [])
              if k in ("threshold", "combine")}
        return (any_of if key == "any_of" else none_of)(qs, **kw)
    raise DSLError(f"unknown assertion type: {item}")


def load_suite(path: str | pathlib.Path) -> Suite:
    data = yaml.safe_load(pathlib.Path(path).read_text())
    cases = []
    for c in data.get("cases", []):
        cases.append(Case(id=c["id"], input=c.get("input"), output=c["output"],
                          assertions=[_parse_assert(a) for a in c.get("assert", [])]))
    return Suite(name=data.get("suite", "suite"), cases=cases,
                 judge=data.get("judge", {}))


def _slug(text: str) -> str:
    s = "".join(ch if ch.isalnum() else "_" for ch in text.lower())
    return "_".join(filter(None, s.split("_")))[:40] or "q"
