"""Torch-free Laya judge for Python: the same decision model the torch backend
runs, driven here through ONNX Runtime with the HF `tokenizers` crate. One forward
pass turns a (state, binary-question) pair into a calibrated P(true).

This is a line-for-line port of the TypeScript judge in js/src/laya.ts, which is
itself a faithful port of laya's own ONNXAgent decode path. All three are checked
against one golden set and agree to 0.0003. No torch, no transformers.
"""
from __future__ import annotations
import glob
import os

import numpy as np

from .base import Judge, JudgeQuestion, JudgeAnswer

# Fixed for the convaiinnovations/laya checkpoint (ModernBERT-large).
CLS, SEP, MASK, PAD = 50281, 50282, 50284, 50283
MAX_LEN = 512
HEAD_MAX_LEN = 192
# temperature_by_options["noul:2"] from the checkpoint config; a noul question has 2 options.
NOUL_TEMP = 1.983399510383606


def _is_laya_tokenizer(tok_json: str) -> bool:
    """True when this tokenizer.json is laya's ModernBERT tokenizer, checked by its
    special-token ids rather than its path, so a different model's tokenizer sitting
    in the same cache cannot pass for it."""
    try:
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(tok_json)
        return (t.token_to_id("[CLS]") == CLS and t.token_to_id("[SEP]") == SEP
                and t.token_to_id("[MASK]") == MASK and t.token_to_id("[PAD]") == PAD)
    except Exception:
        return False


def _resolve_tokenizer(path: str) -> str:
    """Accept a tokenizer.json, a dir holding one, or a models/ tree to search."""
    if os.path.isfile(path):
        return path
    direct = os.path.join(path, "tokenizer.json")
    if os.path.isfile(direct):
        return direct
    hits = sorted(glob.glob(os.path.join(path, "**", "tokenizer.json"), recursive=True))
    if not hits:
        raise FileNotFoundError(f"no tokenizer.json under {path!r}")
    # In a shared HF cache several tokenizer.json files may be present. Keep only the
    # ones that are actually laya's; a stray tokenizer from another model (wrong vocab)
    # would crash the embedding lookup, and it must not win just by sitting shallower.
    laya_hits = [h for h in hits if _is_laya_tokenizer(h)]
    pool = laya_hits or hits
    # Among the matches, prefer the top-level one over nested variants.
    pool.sort(key=lambda p: (p.count(os.sep), len(p)))
    return pool[0]


class OnnxLayaJudge(Judge):
    """Scores noul (binary) questions through laya.onnx. Only noul is supported,
    which is all the injection guardrail uses."""

    def __init__(self, onnx_path: str, tokenizer_path: str):
        from tokenizers import Tokenizer  # no torch
        import onnxruntime as ort

        self.tok = Tokenizer.from_file(_resolve_tokenizer(tokenizer_path))
        # We slice manually, exactly like the TS port, so disable the tokenizer's own
        # truncation/padding to keep the two implementations bit-identical.
        self.tok.no_truncation()
        self.tok.no_padding()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, sess_options=so)

    def _enc(self, text: str, max_len: int | None = None) -> list[int]:
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    # Port of laya.common.build_sequence for a noul question:
    # [CLS] noul question: <ins> [SEP] [MASK] opt0 [MASK] opt1 [SEP] state [SEP]
    def _build_sequence(self, state: str, instructions: str):
        opts = ["false: no", "true: yes"]  # noul {true:"yes", false:"no"} renders like this
        head_ids = self._enc(f"noul question: {instructions}")
        opt_ids = [[MASK] + self._enc(" " + o, 48) for o in opts]

        opt_budget = HEAD_MAX_LEN - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (HEAD_MAX_LEN - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = HEAD_MAX_LEN - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, opt_budget)]

        ids = [CLS] + head_ids + [SEP]
        markers: list[int] = []
        for o in opt_ids:
            markers.append(len(ids))
            ids += o
        ids.append(SEP)

        room = max(0, MAX_LEN - len(ids) - 1)
        st = self._enc(state)[:room]  # truncate_left=False for a string state
        ids = ids + st + [SEP]

        ids = ids[:MAX_LEN]
        markers = [m for m in markers if m < MAX_LEN]
        return ids, markers

    def _score_noul(self, state: str, instructions: list[str]) -> list[float]:
        rows = [self._build_sequence(state, ins) for ins in instructions]
        B = len(rows)
        M = 2  # noul = 2 options
        seq_len = max(len(ids) for ids, _ in rows)

        input_ids = np.full((B, seq_len), PAD, dtype=np.int64)
        attn = np.zeros((B, seq_len), dtype=np.int64)
        mpos = np.zeros((B, M), dtype=np.int64)
        mmask = np.zeros((B, M), dtype=bool)
        qtype = np.full((B,), 2, dtype=np.int64)  # noul

        for b, (ids, markers) in enumerate(rows):
            n = len(ids)
            input_ids[b, :n] = ids
            attn[b, :n] = 1
            for j in range(M):
                mpos[b, j] = markers[j] if j < len(markers) else 0
                mmask[b, j] = True

        out = self.session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attn,
                "marker_pos": mpos,
                "marker_mask": mmask,
                "qtype": qtype,
            },
        )
        logits = out[0]  # [B, M]

        scores = []
        for b in range(B):
            z = logits[b, :2] / NOUL_TEMP
            z = z - z.max()
            e = np.exp(z)
            p = e / e.sum()
            scores.append(float(p[1]))  # P(true) = p[1]
        return scores

    def _require_noul(self, questions: list[JudgeQuestion]) -> None:
        bad = [q.qid for q in questions if q.kind != "noul"]
        if bad:
            raise NotImplementedError(
                f"OnnxLayaJudge scores noul questions only; got non-noul: {bad}"
            )

    def judge(self, state, questions):
        self._require_noul(questions)
        scores = self._score_noul(str(state), [q.instructions for q in questions])
        return [JudgeAnswer(q.qid, "noul", p_true=s)
                for q, s in zip(questions, scores)]

    def judge_batch(self, states, questions, *, batch_size: int = 64):
        self._require_noul(questions)
        return [self.judge(s, questions) for s in states]
