"""Adapters so the scanner fits into any seam: a decorator for any function, a
wrapper for any LLM client callable, and ASGI middleware for the web seam. The CLI
and pre-commit hook cover the pipeline seam."""
from __future__ import annotations
import functools
from typing import Callable

from .core import Finding, get_scanner, Scanner


class InjectionDetected(Exception):
    def __init__(self, finding: Finding):
        self.finding = finding
        super().__init__(f"prompt injection detected (score={finding.score}): "
                         f"{finding.top_question}")


def _scanner(s: Scanner | None) -> Scanner:
    return s or get_scanner()


def guard(arg: int | str = 0, *, on_detect="raise", scanner: Scanner | None = None):
    """Decorator: scan a text argument before the function runs.

    @guard()                 # scan the first positional arg
    @guard("prompt")         # scan the `prompt` kwarg
    on_detect: "raise" (default), "skip" (return None), or a callable(finding, *a, **kw).
    """
    def deco(fn: Callable):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            text = kwargs[arg] if isinstance(arg, str) else args[arg]
            finding = _scanner(scanner).scan(text)
            if finding.is_attack:
                if on_detect == "raise":
                    raise InjectionDetected(finding)
                if on_detect == "skip":
                    return None
                if callable(on_detect):
                    return on_detect(finding, *args, **kwargs)
            return fn(*args, **kwargs)
        return wrapper
    return deco


def wrap_callable(fn: Callable, *, on_detect="raise", scanner: Scanner | None = None):
    """Wrap any callable whose first argument is the user text (e.g. an LLM client
    call). Returns a drop-in replacement that scans first."""
    return guard(0, on_detect=on_detect, scanner=scanner)(fn)


class ASGIGuard:
    """Minimal ASGI middleware: scan the raw request body, reject attacks with a
    400 before they reach your handler. Fits FastAPI / Starlette:

        app.add_middleware(ASGIGuard)
    """
    def __init__(self, app, *, scanner: Scanner | None = None, status: int = 400):
        self.app = app
        self.scanner = scanner
        self.status = status

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            return await self.app(scope, receive, send)

        body = b""
        more = True
        while more:
            msg = await receive()
            body += msg.get("body", b"")
            more = msg.get("more_body", False)

        text = body.decode("utf-8", "ignore")
        if text and _scanner(self.scanner).scan(text).is_attack:
            await send({"type": "http.response.start", "status": self.status,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body",
                        "body": b'{"error":"prompt injection detected"}'})
            return

        async def replay():  # hand the buffered body to the downstream app
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, replay, send)
