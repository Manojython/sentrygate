"""Turn a JudgeAnswer into a PASS / FAIL / INCONCLUSIVE verdict.

noul:  decide at the 0.5 boundary; INCONCLUSIVE only near a coin-flip (margin).
choice/score: INCONCLUSIVE when the modal probability is weak.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from .model import Assertion, Verdict, MIN_MARGIN, CHOICE_MIN_CONF, SCORE_MIN_CONF
from .judge.base import JudgeAnswer


@dataclass
class Result:
    qid: str
    label: str
    verdict: Verdict
    value: Any        # p_true | label | score
    strength: float   # margin (noul) or modal prob (choice/score)


def decide(a: Assertion, ans: JudgeAnswer) -> Result:
    if a.kind == "noul":
        p = ans.p_true
        margin = 2 * abs(p - 0.5)
        if margin < MIN_MARGIN:
            v = Verdict.INCONCLUSIVE
        else:
            present = p >= 0.5
            v = Verdict.PASS if present == a.want_true else Verdict.FAIL
        return Result(a.qid, a.label, v, p, margin)

    if a.kind == "choice":
        mp = ans.max_prob or 0.0
        if mp < CHOICE_MIN_CONF:
            v = Verdict.INCONCLUSIVE
        else:
            v = Verdict.PASS if ans.label == a.want else Verdict.FAIL
        return Result(a.qid, a.label, v, ans.label, mp)

    # score
    mp = ans.max_prob or 0.0
    if mp < SCORE_MIN_CONF:
        v = Verdict.INCONCLUSIVE
    else:
        v = Verdict.PASS if ans.score >= a.floor else Verdict.FAIL
    return Result(a.qid, a.label, v, ans.score, mp)


def decide_ensemble(a: Assertion, sub_p_true: list[float]) -> Result:
    """Combine sub-question P(true) values into one detector verdict.

    A tuned detector, so there is no inconclusive band: the behavior is present
    when the combined score clears `a.threshold`.
    """
    scores = [float(p) for p in sub_p_true]
    combined = max(scores) if a.combine == "max" else sum(scores) / len(scores)
    present = combined >= a.threshold
    v = Verdict.PASS if present == a.want_true else Verdict.FAIL
    return Result(a.qid, a.label, v, combined, combined)
