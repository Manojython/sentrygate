"""Laya-backed judge. Local, ~30 ms/call, free, fine-tunable.

`laya` (and torch) are imported lazily so that importing prompttest does not
require the model stack unless you actually judge.
"""
from __future__ import annotations
from .base import Judge, JudgeQuestion, JudgeAnswer


def _to_laya(questions: list[JudgeQuestion]) -> dict:
    return {q.qid: {"type": q.kind, "instructions": q.instructions,
                    "criteria": q.criteria} for q in questions}


def _map(q: JudgeQuestion, a: dict) -> JudgeAnswer:
    if q.kind == "noul":
        p = a.get("noul")
        if p is None:
            p = (a.get("probabilities") or {}).get("true")
        return JudgeAnswer(q.qid, "noul", p_true=float(p), raw=a)
    if q.kind == "choice":
        return JudgeAnswer(q.qid, "choice", label=a.get("choice"),
                           max_prob=float(a.get("answer_confidence", 0.0)), raw=a)
    return JudgeAnswer(q.qid, "score", score=float(a.get("score")),
                       max_prob=float(a.get("answer_confidence", 0.0)), raw=a)


class LayaJudge(Judge):
    def __init__(self, model: str = "convaiinnovations/laya", *, device=None,
                 compile: bool = False, fast: bool = False, warmup: bool = True):
        import laya  # lazy
        self.agent = laya.load(model, device=device, compile=compile, fast=fast)
        if warmup:
            self.warmup()

    def warmup(self) -> None:
        try:
            self.agent.warmup()
        except Exception:  # noqa: BLE001 — warmup is best-effort
            pass

    def judge(self, state, questions):
        res = self.agent.predict(state, _to_laya(questions))
        return [_map(q, res["answers"][q.qid]) for q in questions]

    def judge_batch(self, states, questions, *, batch_size: int = 64):
        laya_q = _to_laya(questions)
        results = self.agent.predict_batch(states, laya_q, batch_size=batch_size,
                                           sort_by_length=True)
        return [[_map(q, r["answers"][q.qid]) for q in questions] for r in results]
