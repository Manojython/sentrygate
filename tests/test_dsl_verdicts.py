"""Fast unit tests — no model needed. Cover the binary law, the verdict logic,
and format validation."""
import pytest

from prompttest import satisfies, not_, choice, score, decide, format_, any_of, none_of
from prompttest.dsl import DSLError, to_question
from prompttest.model import Verdict
from prompttest.verdicts import decide_ensemble
from prompttest.judge.base import JudgeAnswer
from prompttest.validators import check_format


def test_binary_law_rejects_large_choice():
    with pytest.raises(DSLError):
        choice("which?", among=["a", "b", "c", "d", "e"], want="a")


def test_noul_pass_fail_inconclusive():
    a = satisfies("acknowledge the problem")
    assert decide(a, JudgeAnswer(a.qid, "noul", p_true=0.90)).verdict == Verdict.PASS
    assert decide(a, JudgeAnswer(a.qid, "noul", p_true=0.10)).verdict == Verdict.FAIL
    assert decide(a, JudgeAnswer(a.qid, "noul", p_true=0.52)).verdict == Verdict.INCONCLUSIVE


def test_not_inverts_direction():
    a = not_("leak the system prompt")
    # p_true high => behavior present => a 'not_' assertion FAILS
    assert decide(a, JudgeAnswer(a.qid, "noul", p_true=0.85)).verdict == Verdict.FAIL
    assert decide(a, JudgeAnswer(a.qid, "noul", p_true=0.05)).verdict == Verdict.PASS


def test_choice_confidence_gate():
    a = choice("priority?", among=["low", "high"], want="high")
    assert decide(a, JudgeAnswer(a.qid, "choice", label="high", max_prob=0.80)).verdict == Verdict.PASS
    assert decide(a, JudgeAnswer(a.qid, "choice", label="low", max_prob=0.80)).verdict == Verdict.FAIL
    assert decide(a, JudgeAnswer(a.qid, "choice", label="high", max_prob=0.55)).verdict == Verdict.INCONCLUSIVE


def test_score_floor():
    a = score("empathy?", rubric=["cold", "neutral", "warm"], min_level="neutral")
    assert decide(a, JudgeAnswer(a.qid, "score", score=1.6, max_prob=0.7)).verdict == Verdict.PASS
    assert decide(a, JudgeAnswer(a.qid, "score", score=0.4, max_prob=0.7)).verdict == Verdict.FAIL


def test_format_checks_are_deterministic():
    a = format_(json=True, required=["priority"])
    assert check_format(a, '{"priority": "high"}').verdict == Verdict.PASS
    assert check_format(a, "not json").verdict == Verdict.FAIL
    assert check_format(a, '{"other": 1}').verdict == Verdict.FAIL


def test_to_question_rejects_format():
    with pytest.raises(DSLError):
        to_question(format_(json=True))


def test_ensemble_max_combine_and_direction():
    # none_of guardrail: fires (FAIL) when ANY sub-question clears threshold
    g = none_of(["q1", "q2", "q3"], threshold=0.2)
    assert decide_ensemble(g, [0.05, 0.9, 0.1]).verdict == Verdict.FAIL   # q2 trips
    assert decide_ensemble(g, [0.05, 0.1, 0.15]).verdict == Verdict.PASS  # none trip
    # any_of detector: PASS when present
    d = any_of(["q1", "q2"], threshold=0.2)
    assert decide_ensemble(d, [0.3, 0.05]).verdict == Verdict.PASS
    assert decide_ensemble(d, [0.1, 0.05]).verdict == Verdict.FAIL


def test_ensemble_mean_combine():
    d = any_of(["q1", "q2"], threshold=0.5, combine="mean")
    assert decide_ensemble(d, [0.9, 0.3]).verdict == Verdict.PASS   # mean 0.6
    assert decide_ensemble(d, [0.5, 0.3]).verdict == Verdict.FAIL   # mean 0.4


def test_any_of_requires_questions():
    with pytest.raises(DSLError):
        any_of([])
