"""The judge interface. Backends (Laya today; Jev / two-tier later) implement
this without the DSL or harness needing to change."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class JudgeQuestion:
    qid: str
    kind: Literal["noul", "choice", "score"]
    instructions: str
    criteria: Any  # dict for noul/choice, ordered list for score


@dataclass
class JudgeAnswer:
    qid: str
    kind: str
    p_true: float | None = None      # noul: P(true)
    label: str | None = None         # choice: top label
    score: float | None = None       # score: expected level (float)
    max_prob: float | None = None    # modal probability (choice/score confidence)
    raw: dict = field(default_factory=dict)


class Judge(ABC):
    @abstractmethod
    def judge(self, state: str | dict, questions: list[JudgeQuestion]) -> list[JudgeAnswer]:
        ...

    @abstractmethod
    def judge_batch(self, states: list[str | dict], questions: list[JudgeQuestion],
                    *, batch_size: int = 64) -> list[list[JudgeAnswer]]:
        ...

    def warmup(self) -> None:
        pass
