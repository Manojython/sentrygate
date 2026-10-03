"""`prompttest test <suite.yaml>` — run a contract, print a table, exit non-zero
on any FAIL (so it gates CI)."""
from __future__ import annotations
import argparse
import os
import pathlib
import sys

from .model import Verdict
from .dsl import load_suite


def _models_dir() -> str:
    return str(pathlib.Path.cwd() / "models")


def _print_case(cr):
    print(f"\n### {cr.id}")
    print(f"  {'id':20s} {'verdict':13s} {'value':>8} {'strength':>9}  assertion")
    print("  " + "-" * 82)
    for r in cr.results:
        vs = f"{r.value:.2f}" if isinstance(r.value, float) else str(r.value)[:8]
        print(f"  {r.qid[:20]:20s} {r.verdict.value:13s} {vs:>8} "
              f"{r.strength:>9.2f}  {r.label}")


def cmd_test(args) -> int:
    # keep model/data downloads inside the project unless the user set HF_HOME
    os.environ.setdefault("HF_HOME", _models_dir())
    os.environ.setdefault("HF_HUB_CACHE", _models_dir())

    suite = load_suite(args.suite)
    from .judge.laya_judge import LayaJudge  # lazy: needs laya+torch
    model = suite.judge.get("model", "convaiinnovations/laya")
    judge = LayaJudge(model)

    from .harness import run_suite
    results = run_suite(judge, suite)

    n_fail = 0
    for cr in results:
        _print_case(cr)
        n_fail += sum(1 for r in cr.results if r.verdict == Verdict.FAIL)

    total = sum(len(cr.results) for cr in results)
    inconclusive = sum(1 for cr in results for r in cr.results
                       if r.verdict == Verdict.INCONCLUSIVE)
    print(f"\n{total} assertions: {total - n_fail - inconclusive} pass, "
          f"{n_fail} fail, {inconclusive} inconclusive")
    return 1 if n_fail else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="prompttest")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("test", help="run a YAML suite against the judge")
    t.add_argument("suite", help="path to a suite .yaml")
    t.set_defaults(func=cmd_test)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
