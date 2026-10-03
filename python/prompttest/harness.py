"""Run a contract over (input, output) pairs: batch the judged assertions in one
pass, run format assertions deterministically, collect verdicts."""
from __future__ import annotations
from dataclasses import dataclass, field

from .model import Assertion, Verdict
from .dsl import Suite, Case, to_question
from .verdicts import Result, decide, decide_ensemble
from .validators import check_format
from .judge.base import Judge, JudgeQuestion


def make_state(user_input, model_output: str) -> str:
    """Readable state so zero-shot questions can reference input and output."""
    if user_input is None:
        return str(model_output)
    return f"=== USER INPUT ===\n{user_input}\n\n=== ASSISTANT OUTPUT ===\n{model_output}"


def _ensemble_questions(a: Assertion) -> list[JudgeQuestion]:
    return [JudgeQuestion(f"{a.qid}#{i}", "noul", q, {"true": "yes", "false": "no"})
            for i, q in enumerate(a.subs)]


def judge_pair(judge: Judge, user_input, output: str,
               assertions: list[Assertion]) -> list[Result]:
    results: dict[str, Result] = {}
    questions: list[JudgeQuestion] = []

    for a in assertions:
        if a.kind == "format":
            results[a.qid] = check_format(a, output)        # deterministic, no judge
        elif a.kind == "ensemble":
            questions.extend(_ensemble_questions(a))
        else:
            questions.append(to_question(a))

    if questions:
        answers = judge.judge(make_state(user_input, output), questions)
        by_id = {ans.qid: ans for ans in answers}
        for a in assertions:
            if a.kind == "ensemble":
                subs = [by_id[f"{a.qid}#{i}"].p_true for i in range(len(a.subs))]
                results[a.qid] = decide_ensemble(a, subs)
            elif a.kind != "format":
                results[a.qid] = decide(a, by_id[a.qid])

    return [results[a.qid] for a in assertions]


@dataclass
class CaseResult:
    id: str
    results: list[Result] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(r.verdict == Verdict.FAIL for r in self.results)


def run_suite(judge: Judge, suite: Suite) -> list[CaseResult]:
    out = []
    for case in suite.cases:
        out.append(CaseResult(case.id, judge_pair(judge, case.input, case.output,
                                                  case.assertions)))
    return out
