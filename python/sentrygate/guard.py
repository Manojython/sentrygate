"""A small composable pipeline: ordered detectors (which may block) and
transforms (which rewrite the text) that run over a prompt before it leaves the
machine.

The scanner engine is wrapped by `InjectionDetector`, which imports it lazily so
that importing sentrygate never loads the decision model. A PII-only `Guard` runs
with no model and no torch/onnx installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from . import pii


@runtime_checkable
class Detector(Protocol):
    def check(self, text: str):
        """Return a Finding-like object with an `is_attack` attribute."""
        ...


@runtime_checkable
class Transform(Protocol):
    def apply(self, text: str):
        """Return (new_text, info) where info is a dict (e.g. a mapping)."""
        ...


class InjectionDetector:
    """Thin, lazy wrapper over the scanner engine. The scanner (and the model) is
    only imported/loaded the first time `check` runs, not when sentrygate is imported."""

    def __init__(self, **scanner_kwargs):
        self._kwargs = scanner_kwargs
        self._scanner = None

    def _get(self):
        if self._scanner is None:
            from . import scanner  # lazy: pulls in the model backend only on use
            self._scanner = scanner.get_scanner(**self._kwargs)
        return self._scanner

    def check(self, text: str):
        return self._get().scan(text)


class PIIMasker:
    """Transform that masks PII via `pii.mask`. `types` restricts which entity
    types are masked (None = all)."""

    def __init__(self, types: set | None = None):
        self.types = types

    def apply(self, text: str):
        result = pii.mask(text, types=self.types)
        return result.masked_text, {"mapping": result.mapping, "spans": result.spans}


@dataclass
class GuardResult:
    allowed: bool
    text: str
    findings: list = field(default_factory=list)
    mapping: dict = field(default_factory=dict)
    blocked_reason: str | None = None


class Guard:
    """Run an ordered list of steps over a prompt.

    Detectors run first; if any flags an attack the prompt is blocked and the
    remaining steps are skipped. Transforms then run in order, accumulating a
    reversible mapping and updating the text.
    """

    def __init__(self, steps: list | None = None):
        self.steps = list(steps or [])
        self._mapping: dict = {}   # accumulated by the last process() call

    def process(self, text: str) -> GuardResult:
        findings = []
        detectors = [s for s in self.steps if hasattr(s, "check")]
        transforms = [s for s in self.steps if hasattr(s, "apply")]

        for det in detectors:
            finding = det.check(text)
            findings.append(finding)
            if getattr(finding, "is_attack", False):
                reason = getattr(finding, "top_question", None) or "injection detected"
                self._mapping = {}
                return GuardResult(allowed=False, text=text, findings=findings,
                                   mapping={}, blocked_reason=reason)

        mapping: dict = {}
        for tf in transforms:
            text, info = tf.apply(text)
            if info and "mapping" in info:
                mapping.update(info["mapping"])

        self._mapping = mapping
        return GuardResult(allowed=True, text=text, findings=findings,
                           mapping=mapping, blocked_reason=None)

    def unmask(self, text: str, mapping: dict | None = None) -> str:
        """Reverse a mapping, defaulting to the one accumulated by the last
        process() call."""
        return pii.unmask(text, mapping if mapping is not None else self._mapping)
