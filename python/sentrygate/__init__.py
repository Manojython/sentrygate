"""sentrygate — an on-device layer that sits between an app and an LLM and cleans a
prompt before it leaves the machine. It inspects the prompt for injection and
redacts sensitive data locally, so secrets and PII never egress. Injection
detection is handled by the bundled ``scanner`` engine, loaded lazily and only when
an injection step is actually used; redaction runs on deterministic, stdlib-only
PII recognizers today and needs no model at all. NER-based detection and a Laya
policy layer are optional hooks, not yet wired in by default.
"""
from __future__ import annotations

from .pii import detect, mask, unmask, MaskResult, PIISpan
from .guard import Guard, GuardResult, PIIMasker, InjectionDetector

__version__ = "0.1.1"

__all__ = [
    "detect", "mask", "unmask", "MaskResult", "PIISpan",
    "Guard", "GuardResult", "PIIMasker", "InjectionDetector",
    "redact", "scan", "is_attack", "guard_prompt",
    "guard", "wrap_callable", "ASGIGuard", "InjectionDetected",
    "Scanner", "Finding", "get_scanner",
]

# The injection scanner (and its model) is loaded lazily: importing sentrygate, and
# the whole PII path, must stay free of the model and of onnx/torch. These names are
# served from sentrygate.scanner only when first accessed.
_LAZY = {
    "Scanner", "Finding", "get_scanner", "is_attack",
    "guard", "wrap_callable", "InjectionDetected", "ASGIGuard",
}


def __getattr__(name: str):
    if name in _LAZY:
        from . import scanner
        return getattr(scanner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def redact(text: str, types: set | None = None) -> MaskResult:
    """Headline on-device feature: mask PII locally, no model required."""
    return mask(text, types=types)


def scan(text: str, **kwargs):
    """Scan a prompt for injection/jailbreak. Loads the scanner engine lazily."""
    from . import scanner
    return scanner.scan(text, **kwargs)


def guard_prompt(text: str, mask_pii: bool = True, detect_injection: bool = True,
                 **scanner_kwargs) -> GuardResult:
    """Build a Guard with the requested steps and run it over `text`.

    Detectors run before transforms, so injection is checked before any
    redaction. The scanner engine is only loaded when `detect_injection` is True.
    """
    steps = []
    if detect_injection:
        steps.append(InjectionDetector(**scanner_kwargs))
    if mask_pii:
        steps.append(PIIMasker())
    return Guard(steps).process(text)
