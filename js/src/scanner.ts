// The detection engine: ask several targeted binary questions, flag the text
// when the combined (MAX) score clears a tuned threshold. Mirrors the Python
// Scanner exactly, with the same questions, threshold and Finding shape.

import { LayaOnnxJudge, Question } from "./laya.js";
import { INJECTION_QUESTIONS, INJECTION_THRESHOLD } from "./presets.js";
import { resolveModel } from "./model.js";

export interface Finding {
  isAttack: boolean;
  score: number; // combined (MAX) P(true) across questions
  threshold: number;
  topQuestion: string; // which question fired hardest
  perQuestion: Record<string, number>;
  preview: string;
}

export interface ScannerOptions {
  questions?: string[];
  threshold?: number;
  modelDir?: string;
  onnxPath?: string;
}

export class Scanner {
  private judge = new LayaOnnxJudge();
  private ready: Promise<void> | null = null;
  private questions: string[];
  private threshold: number;
  private q: Question[];
  private modelDir?: string;
  private onnxPath?: string;

  constructor(opts: ScannerOptions = {}) {
    this.questions = opts.questions ?? INJECTION_QUESTIONS;
    this.threshold = opts.threshold ?? INJECTION_THRESHOLD;
    this.modelDir = opts.modelDir;
    this.onnxPath = opts.onnxPath;
    this.q = this.questions.map((instructions, i) => ({ id: `inj#${i}`, instructions }));
  }

  private init(): Promise<void> {
    if (!this.ready) {
      this.ready = resolveModel(this.modelDir, this.onnxPath).then(({ modelDir, onnxPath }) =>
        this.judge.init(modelDir, onnxPath),
      );
    }
    return this.ready;
  }

  private finding(text: string, scores: number[]): Finding {
    let top = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[top]) top = i;
    const perQuestion: Record<string, number> = {};
    this.questions.forEach((qq, i) => (perQuestion[qq] = round(scores[i])));
    return {
      isAttack: scores[top] >= this.threshold,
      score: round(scores[top]),
      threshold: this.threshold,
      topQuestion: this.questions[top],
      perQuestion,
      preview: text.slice(0, 120),
    };
  }

  async scan(text: string): Promise<Finding> {
    await this.init();
    const scores = await this.judge.scoreNoul(text, this.q);
    return this.finding(text, scores);
  }

  async scanBatch(texts: string[]): Promise<Finding[]> {
    await this.init();
    const out: Finding[] = [];
    for (const t of texts) out.push(this.finding(t, await this.judge.scoreNoul(t, this.q)));
    return out;
  }
}

function round(x: number): number {
  return Math.round(x * 1000) / 1000;
}

let _default: Scanner | null = null;

export function getScanner(opts: ScannerOptions = {}): Scanner {
  if (Object.keys(opts).length) return new Scanner(opts);
  if (!_default) _default = new Scanner();
  return _default;
}
