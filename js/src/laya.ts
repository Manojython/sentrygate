// Torch-free Laya judge: the same decision model the Python package runs, driven
// here through ONNX Runtime with a pure-JS tokenizer. One forward pass turns a
// (state, binary-question) pair into a calibrated P(true). This is a faithful port
// of laya's own ONNXAgent decode path, validated against it bit-for-bit in the tests.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import * as ort from "onnxruntime-node";
import { PreTrainedTokenizer } from "@huggingface/transformers";

// Fixed for the convaiinnovations/laya checkpoint (ModernBERT-large).
const SPECIAL = { cls: 50281, sep: 50282, mask: 50284, pad: 50283 };
const MAX_LEN = 512;
const HEAD_MAX_LEN = 192;
// temperature_by_options["noul:2"] from the checkpoint config; a noul question has 2 options.
const NOUL_TEMP = 1.983399510383606;

export interface Question {
  id: string;
  instructions: string;
}

// Render noul options exactly as laya does for criteria {true:"yes", false:"no"}.
function noulOptions(): [string, string] {
  return ["false: no", "true: yes"];
}

export class LayaOnnxJudge {
  private tok!: PreTrainedTokenizer;
  private session!: ort.InferenceSession;

  /** `modelDir` holds tokenizer.json + tokenizer_config.json; `onnxPath` is laya.onnx. */
  async init(modelDir: string, onnxPath: string): Promise<void> {
    const tokJSON = JSON.parse(readFileSync(join(modelDir, "tokenizer.json"), "utf8"));
    let tokCfg: any = {};
    try {
      tokCfg = JSON.parse(readFileSync(join(modelDir, "tokenizer_config.json"), "utf8"));
    } catch {
      /* optional */
    }
    this.tok = new PreTrainedTokenizer(tokJSON, tokCfg);
    this.session = await ort.InferenceSession.create(onnxPath, {
      graphOptimizationLevel: "all",
    });
  }

  private enc(text: string, maxLen?: number): number[] {
    let ids = this.tok.encode(text, { add_special_tokens: false }) as number[];
    if (maxLen && ids.length > maxLen) ids = ids.slice(0, maxLen);
    return ids;
  }

  // Port of laya.common.build_sequence for a noul question:
  // [CLS] noul question: <ins> [SEP] [MASK] opt0 [MASK] opt1 [SEP] state [SEP]
  private buildSequence(state: string, instructions: string): { ids: number[]; markers: number[] } {
    const { cls, sep, mask } = SPECIAL;
    const opts = noulOptions();
    let headIds = this.enc(`noul question: ${instructions}`);
    let optIds = opts.map((o) => [mask, ...this.enc(" " + o, 48)]);

    let optBudget = HEAD_MAX_LEN - optIds.reduce((s, o) => s + o.length, 0);
    if (optBudget < 16) {
      const per = Math.max(4, Math.floor((HEAD_MAX_LEN - 16) / Math.max(1, optIds.length)));
      optIds = optIds.map((o) => o.slice(0, per));
      optBudget = HEAD_MAX_LEN - optIds.reduce((s, o) => s + o.length, 0);
    }
    headIds = headIds.slice(0, Math.max(8, optBudget));

    let ids = [cls, ...headIds, sep];
    const markers: number[] = [];
    for (const o of optIds) {
      markers.push(ids.length);
      ids.push(...o);
    }
    ids.push(sep);

    const room = Math.max(0, MAX_LEN - ids.length - 1);
    const stateIds = this.enc(state);
    const st = stateIds.slice(0, room); // truncate_left=false for a string state
    ids = [...ids, ...st, sep];

    ids = ids.slice(0, MAX_LEN);
    return { ids, markers: markers.filter((m) => m < MAX_LEN) };
  }

  /** Score several noul questions against one state in a single session run; returns P(true) each. */
  async scoreNoul(state: string, questions: Question[]): Promise<number[]> {
    const rows = questions.map((q) => this.buildSequence(state, q.instructions));
    const B = rows.length;
    const M = 2; // noul = 2 options
    const seqLen = Math.max(...rows.map((r) => r.ids.length));

    const inputIds = new BigInt64Array(B * seqLen);
    const attn = new BigInt64Array(B * seqLen);
    const mpos = new BigInt64Array(B * M);
    const mmask = new Uint8Array(B * M);
    const qtype = new BigInt64Array(B);

    for (let b = 0; b < B; b++) {
      const r = rows[b];
      for (let i = 0; i < seqLen; i++) {
        const tokenId = i < r.ids.length ? r.ids[i] : SPECIAL.pad;
        inputIds[b * seqLen + i] = BigInt(tokenId);
        attn[b * seqLen + i] = i < r.ids.length ? 1n : 0n;
      }
      for (let j = 0; j < M; j++) {
        mpos[b * M + j] = BigInt(r.markers[j] ?? 0);
        mmask[b * M + j] = 1;
      }
      qtype[b] = 2n; // noul
    }

    const feeds: Record<string, ort.Tensor> = {
      input_ids: new ort.Tensor("int64", inputIds, [B, seqLen]),
      attention_mask: new ort.Tensor("int64", attn, [B, seqLen]),
      marker_pos: new ort.Tensor("int64", mpos, [B, M]),
      marker_mask: new ort.Tensor("bool", mmask, [B, M]),
      qtype: new ort.Tensor("int64", qtype, [B]),
    };

    const out = await this.session.run(feeds);
    const logits = out["logits"].data as Float32Array; // [B, M]

    const scores: number[] = [];
    for (let b = 0; b < B; b++) {
      const z0 = logits[b * M + 0] / NOUL_TEMP;
      const z1 = logits[b * M + 1] / NOUL_TEMP;
      const mx = Math.max(z0, z1);
      const e0 = Math.exp(z0 - mx);
      const e1 = Math.exp(z1 - mx);
      scores.push(e1 / (e0 + e1)); // P(true) = p[1]
    }
    return scores;
  }
}
