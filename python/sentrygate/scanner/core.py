"""The detection engine: a Scanner that asks several targeted binary questions
and flags text when the combined (MAX) score clears a tuned threshold.

Questions and threshold come from prompttest's measured injection preset
(AUC ~0.87 on deepset/prompt-injections). Swap `questions`/`threshold` or pass
a fine-tuned judge to specialize.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

from prompttest.presets import INJECTION_QUESTIONS, INJECTION_THRESHOLD
from prompttest.judge.base import Judge, JudgeQuestion


_HF_REPO = "mnjkshrm/sentrygate-laya"   # int8 ONNX export of Laya

_BACKEND_HELP = (
    "the injection guard needs onnxruntime, and it is not importable here.\n"
    "  pip install sentrygate   installs it, and the model fetches on first scan.\n"
    "Already have a laya.onnx? point SENTRYGATE_ONNX at it.\n"
    "More at https://github.com/Manojython/sentrygate"
)


def _bundled_tokenizer() -> str | None:
    """The Laya tokenizer ships inside the package, so only the weights ever need
    fetching. Returns its path, or None if it is somehow missing from the install."""
    p = os.path.join(os.path.dirname(__file__), "data", "tokenizer.json")
    return p if os.path.isfile(p) else None


def _onnx_judge(onnx: str, tok: str | None = None) -> Judge:
    try:
        from prompttest.judge.onnx_judge import OnnxLayaJudge
    except ImportError as e:
        raise ImportError(_BACKEND_HELP) from e
    if tok is None:
        # The tokenizer that ships in the package is the known-good one, so it wins
        # over HF_HOME (a cache dir that may hold no tokenizer.json at all).
        tok = (os.environ.get("SENTRYGATE_MODEL_DIR")
               or _bundled_tokenizer()
               or os.environ.get("HF_HOME")
               or os.path.dirname(onnx) or "models")
    return OnnxLayaJudge(onnx, tok)


def _fetch_laya_onnx():
    """Pull the exported graph from the Hub into ./models, once, the way ollama pulls a
    model on first use. The tokenizer is bundled, so only the weights download. Returns
    the onnx path, or None when onnxruntime is missing; a failed download raises with
    the manual steps so you are never left guessing."""
    try:
        import onnxruntime  # noqa: F401  (no point pulling 440 MB with no runtime for it)
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    import sys
    repo = os.environ.get("SENTRYGATE_HF_REPO") or _HF_REPO
    dest = os.path.join(os.getcwd(), "models")
    print(f"sentrygate: fetching the guard model (~440 MB, first run only) from {repo} ...",
          file=sys.stderr)
    try:
        onnx = hf_hub_download(repo, "laya.onnx", local_dir=dest)
    except Exception as e:
        raise RuntimeError(
            f"sentrygate could not download the model from {repo} ({e}).\n"
            "Pull it by hand and sentrygate picks it up on its own:\n"
            f"  hf download {repo} laya.onnx --local-dir ./models\n"
            '  export SENTRYGATE_ONNX="$PWD/models/laya.onnx"'
        ) from e
    return onnx


def _default_judge(model: str) -> Judge:
    """Pick a backend, in order: a graph you point at (SENTRYGATE_ONNX or a local
    models/laya.onnx), then the torch-backed LayaJudge if torch is installed, then
    the torch-free path, fetching the exported graph from the Hub into ./models. If
    nothing is installed, say so clearly instead of leaking an import error."""
    onnx = os.environ.get("SENTRYGATE_ONNX")
    if not onnx:
        local = os.path.join(os.getcwd(), "models", "laya.onnx")
        if os.path.exists(local):
            onnx = local
    if onnx:
        return _onnx_judge(onnx)

    try:
        from prompttest.judge.laya_judge import LayaJudge  # needs torch
        return LayaJudge(model)
    except ImportError:
        pass

    fetched = _fetch_laya_onnx()
    if fetched:
        return _onnx_judge(fetched)

    raise ImportError(_BACKEND_HELP)


@dataclass
class Finding:
    is_attack: bool
    score: float                    # combined (MAX) P(true) across questions
    threshold: float
    top_question: str               # which question fired hardest
    per_question: dict = field(default_factory=dict)
    preview: str = ""

    def __bool__(self) -> bool:      # `if sentrygate.scan(text): ...`
        return self.is_attack


class Scanner:
    def __init__(self, judge: Judge | None = None, *,
                 questions: list[str] = INJECTION_QUESTIONS,
                 threshold: float = INJECTION_THRESHOLD,
                 model: str = "convaiinnovations/laya"):
        if judge is None:
            judge = _default_judge(model)
        self.judge = judge
        self.questions = list(questions)
        self.threshold = threshold
        self._q = [JudgeQuestion(f"inj#{i}", "noul", q, {"true": "yes", "false": "no"})
                   for i, q in enumerate(self.questions)]

    def _finding(self, text: str, answers) -> Finding:
        scores = [a.p_true for a in answers]
        top = int(max(range(len(scores)), key=lambda i: scores[i]))
        return Finding(is_attack=scores[top] >= self.threshold,
                       score=round(scores[top], 3), threshold=self.threshold,
                       top_question=self.questions[top],
                       per_question={self.questions[i]: round(s, 3)
                                     for i, s in enumerate(scores)},
                       preview=text[:120])

    def scan(self, text: str) -> Finding:
        return self._finding(text, self.judge.judge(text, self._q))

    def scan_batch(self, texts: list[str]) -> list[Finding]:
        results = self.judge.judge_batch(texts, self._q)
        return [self._finding(t, a) for t, a in zip(texts, results)]


_default: Scanner | None = None


def get_scanner(**kwargs) -> Scanner:
    """Lazily build and reuse a module-level scanner (loads the model once)."""
    global _default
    if _default is None or kwargs:
        s = Scanner(**kwargs)
        if not kwargs:
            _default = s
        return s
    return _default


def scan(text: str, **kwargs) -> Finding:
    return get_scanner(**kwargs).scan(text)


def is_attack(text: str, **kwargs) -> bool:
    return scan(text, **kwargs).is_attack
