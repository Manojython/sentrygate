// sentrygate, a local on-device guard layer between your app and an LLM.
// It does two jobs: catch prompt injections (a 421M decision model run in-process
// via ONNX Runtime) and detect and mask PII before a prompt
// leaves the machine (pure regex, no model). No API calls, nothing leaves your
// machine. Same engine and same scores as the Python package.
//
//   import { scan, redact } from "sentrygate";
//
//   if ((await scan(userMessage)).isAttack) reject();   // injection (fetches the model on first use)
//   const safe = redact(userMessage);                    // PII masking (no model)

export { Scanner, getScanner } from "./scanner.js";
export type { Finding, ScannerOptions } from "./scanner.js";
export { guard, guardMiddleware, InjectionDetected } from "./adapters.js";
export type { MiddlewareOptions } from "./adapters.js";
export { INJECTION_QUESTIONS, INJECTION_THRESHOLD } from "./presets.js";

// PII detection and reversible masking (no model required).
export { detect, mask, unmask, redact } from "./pii.js";
export type { PIISpan, MaskResult } from "./pii.js";

import { getScanner, Finding, ScannerOptions } from "./scanner.js";

/** Scan a single piece of text for a prompt injection. Resolves to a Finding (truthy-ish via `.isAttack`). */
export async function scan(text: string, opts?: ScannerOptions): Promise<Finding> {
  return getScanner(opts ?? {}).scan(text);
}

/** Convenience boolean: is this text an injection / jailbreak attempt? */
export async function isAttack(text: string, opts?: ScannerOptions): Promise<boolean> {
  return (await scan(text, opts)).isAttack;
}
