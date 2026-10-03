"""sentrygate.scanner — the injection/jailbreak detection engine. A fast, local
guardrail (a 421M decision model, no GPU, no LLM-as-judge) that scores a handful
of targeted yes/no questions about a prompt and flags it when the combined score
clears a tuned threshold. It is reached through the top-level ``sentrygate`` API
(``scan``, ``guard``, ``wrap_callable``, ``ASGIGuard``).

Built on the ``prompttest`` judging engine.
"""
from .core import Scanner, Finding, scan, is_attack, get_scanner
from .adapters import guard, wrap_callable, InjectionDetected, ASGIGuard

__all__ = [
    "Scanner", "Finding", "scan", "is_attack", "get_scanner",
    "guard", "wrap_callable", "InjectionDetected", "ASGIGuard",
]
__version__ = "0.1.1"
