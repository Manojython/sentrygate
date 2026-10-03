"""sentrygate command line: redact PII locally, or scan a prompt for injection.

    echo "email ana@acme.com" | sentrygate redact
    echo "ignore all previous instructions" | sentrygate scan   # exit 1 on attack
    sentrygate scan prompts/*.txt                                # scan files, exit 1 if any flagged
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import sys


def _pin_downloads_local():
    models = str(pathlib.Path.cwd() / "models")
    os.environ.setdefault("HF_HOME", models)
    os.environ.setdefault("HF_HUB_CACHE", models)


def _read_input(args) -> str:
    if args.text:
        return " ".join(args.text)
    return sys.stdin.read()


def _cmd_redact(args) -> int:
    from . import redact
    r = redact(_read_input(args))
    out = r.masked_text
    sys.stdout.write(out if out.endswith("\n") else out + "\n")
    if args.show_map and r.mapping:
        for ph, val in r.mapping.items():
            print(f"{ph}\t{val}", file=sys.stderr)
    return 0


def _scan_inputs(args) -> list[tuple[str, str]]:
    """Positional args that are all existing files are read as files (so the
    pre-commit hook and `sentrygate scan file.txt` work); otherwise the args are
    taken as literal text, and with no args at all the text is read from stdin."""
    paths = args.text
    if paths and all(os.path.isfile(p) for p in paths):
        out = []
        for p in paths:
            try:
                out.append((p, pathlib.Path(p).read_text(errors="ignore")))
            except OSError as e:
                print(f"sentrygate: cannot read {p}: {e}", file=sys.stderr)
        return out
    if paths:
        return [("<arg>", " ".join(paths))]
    return [("<stdin>", sys.stdin.read())]


def _cmd_scan(args) -> int:
    _pin_downloads_local()
    from . import scan
    findings = [(name, scan(text)) for name, text in _scan_inputs(args)]
    flagged = [(n, f) for n, f in findings if f.is_attack]

    if args.json:
        print(json.dumps([{"source": n, "is_attack": f.is_attack, "score": f.score,
                           "threshold": f.threshold, "top_question": f.top_question,
                           "preview": f.preview} for n, f in findings], indent=2))
    else:
        for n, f in findings:
            label = "ATTACK" if f.is_attack else "clean"
            print(f"[{label}] score={f.score:.2f}  {n}  {f.top_question}", file=sys.stderr)

    return 1 if flagged else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="sentrygate",
        description="Catch injections and mask private data before a prompt leaves your machine.")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("redact", help="mask PII in text from stdin or args (no model)")
    pr.add_argument("text", nargs="*", help="text to redact; omit to read stdin")
    pr.add_argument("--show-map", action="store_true",
                    help="print the placeholder to value map on stderr")
    pr.set_defaults(func=_cmd_redact)

    ps = sub.add_parser("scan",
                        help="scan text, files or stdin for prompt injection (exit 1 on attack)")
    ps.add_argument("text", nargs="*", help="literal text, or file paths; omit to read stdin")
    ps.add_argument("--json", action="store_true", help="machine-readable output")
    ps.set_defaults(func=_cmd_scan)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
