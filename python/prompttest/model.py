"""Core data types and verdict thresholds.

Thresholds were chosen against Laya's observed output semantics (confidence is
derived from the probability, not an independent calibration signal), so we
gate `noul` verdicts on margin from 0.5 rather than an absolute confidence.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# noul: a verdict is INCONCLUSIVE only when P(true) is within this margin of a
# coin-flip, i.e. 2*|p-0.5| < MIN_MARGIN.
MIN_MARGIN = 0.30
# choice / score: modal probability below these -> INCONCLUSIVE.
CHOICE_MIN_CONF = 0.60
SCORE_MIN_CONF = 0.50


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class Assertion:
    """One behavioral claim about an output. Compiles to a single Laya question.

    kind:
      - "noul"   binary; `want_true` is whether we want the behavior present
      - "choice" few-option (<=4); `want` is the required label
      - "score"  ordinal; `floor` is the minimum acceptable level index
      - "format" deterministic (no judge); `spec` holds the checks
    """
    qid: str
    kind: str
    label: str
    instructions: str = ""
    criteria: Any = None
    want_true: bool = True        # noul / ensemble
    want: Any = None              # choice
    floor: int = 0                # score
    rubric: list = field(default_factory=list)  # score (for display)
    spec: dict = field(default_factory=dict)     # format
    subs: list = field(default_factory=list)     # ensemble: sub-question instructions
    combine: str = "max"          # ensemble: how to combine sub-scores
    threshold: float = 0.5        # ensemble: detection cutoff on the combined score
