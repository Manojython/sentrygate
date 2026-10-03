// Thin adapters so the injection guard drops into whatever seam you already
// have: a direct call, a function wrapper, or Express/Connect middleware. Same
// idea as the Python decorator / ASGI middleware.

import { Scanner, Finding, getScanner } from "./scanner.js";

export class InjectionDetected extends Error {
  finding: Finding;
  constructor(finding: Finding) {
    super(`prompt injection detected (score ${finding.score} >= ${finding.threshold})`);
    this.name = "InjectionDetected";
    this.finding = finding;
  }
}

/** Wrap any async function whose first string argument should be scanned first. */
export function guard<A extends any[], R>(
  fn: (...args: A) => Promise<R>,
  opts: { scanner?: Scanner; argIndex?: number } = {},
): (...args: A) => Promise<R> {
  const scanner = opts.scanner ?? getScanner();
  const idx = opts.argIndex ?? 0;
  return async (...args: A): Promise<R> => {
    const text = args[idx];
    if (typeof text === "string") {
      const f = await scanner.scan(text);
      if (f.isAttack) throw new InjectionDetected(f);
    }
    return fn(...args);
  };
}

export interface MiddlewareOptions {
  scanner?: Scanner;
  /** Pull the text to scan out of the request. Default: req.body.prompt | req.body.message | req.body.input. */
  extract?: (req: any) => string | undefined;
  /** HTTP status to respond with when an attack is detected. Default 400. */
  status?: number;
}

/** Express / Connect middleware: reject a request whose payload looks like an injection. */
export function guardMiddleware(opts: MiddlewareOptions = {}) {
  const scanner = opts.scanner ?? getScanner();
  const status = opts.status ?? 400;
  const extract =
    opts.extract ??
    ((req: any) => req?.body?.prompt ?? req?.body?.message ?? req?.body?.input);

  return async (req: any, res: any, next: any) => {
    try {
      const text = extract(req);
      if (typeof text === "string" && text.length) {
        const finding = await scanner.scan(text);
        if (finding.isAttack) {
          res.status(status).json({ error: "prompt injection detected", finding });
          return;
        }
        req.sentrygate = finding;
      }
      next();
    } catch (err) {
      next(err);
    }
  };
}
