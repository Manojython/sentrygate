"""prompttest — a behavioral contract framework for LLM prompts.

Declare assertions about what a prompt's output should do, then have a fast
local decision model (Laya) judge each output. The same contract runs as a CI
gate and as a runtime monitor.
"""
from .model import Assertion, Verdict, MIN_MARGIN, CHOICE_MIN_CONF, SCORE_MIN_CONF
from .dsl import (satisfies, not_, choice, score, format_, any_of, none_of,
                  load_suite, Suite, Case)
from .verdicts import decide, decide_ensemble
from .harness import make_state, judge_pair, run_suite
from .judge.base import Judge, JudgeQuestion, JudgeAnswer
from . import presets

__all__ = [
    "Assertion", "Verdict", "MIN_MARGIN", "CHOICE_MIN_CONF", "SCORE_MIN_CONF",
    "satisfies", "not_", "choice", "score", "format_", "any_of", "none_of",
    "load_suite", "Suite", "Case",
    "decide", "decide_ensemble", "make_state", "judge_pair", "run_suite",
    "Judge", "JudgeQuestion", "JudgeAnswer", "presets",
]
__version__ = "0.1.0"
