// Locate the model assets, fetching the weights on first use the way ollama
// pulls a model. Resolution order for the ONNX graph:
//   1. explicit argument
//   2. the SENTRYGATE_ONNX environment variable
//   3. a ./models/laya.onnx checkout in the current working directory
//   4. otherwise download it once from the Hub into ./models
//
// The tokenizer ships inside the package (data/tokenizer.json), so only the
// weights ever fetch. laya.onnx is ~440 MB (int8). Nothing is sent anywhere at
// inference time.

import { existsSync, mkdirSync, createWriteStream } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

const DEFAULT_HF_REPO = "mnjkshrm/sentrygate-laya"; // int8 ONNX export of Laya

// The tokenizer bundled with this package (ships via the "data" files entry).
function bundledTokenizerDir(): string | null {
  const here = dirname(fileURLToPath(import.meta.url));
  // src/model.ts at dev time, dist/model.js once built: both sit one level
  // below the package root, next to data/.
  const dir = join(here, "..", "data");
  return existsSync(join(dir, "tokenizer.json")) ? dir : null;
}

function resolveTokenizerDir(modelDir: string | undefined, onnxPath: string): string {
  const md =
    modelDir ??
    process.env.SENTRYGATE_MODEL_DIR ??
    bundledTokenizerDir() ??
    dirname(onnxPath);
  if (!existsSync(join(md, "tokenizer.json"))) {
    throw new Error(
      "sentrygate: could not find the tokenizer. The package ships one in data/; " +
        "set SENTRYGATE_MODEL_DIR to a directory containing tokenizer.json, or pass " +
        "{ modelDir } to the Scanner.",
    );
  }
  return md;
}

/**
 * Pull the exported graph from the Hub into ./models, once, the way ollama pulls
 * a model on first use. The tokenizer is bundled, so only the weights download.
 * A failed download throws with the manual steps so you are never left guessing.
 */
async function fetchLayaOnnx(): Promise<string> {
  const repo = process.env.SENTRYGATE_HF_REPO ?? DEFAULT_HF_REPO;
  const dest = join(process.cwd(), "models");
  const out = join(dest, "laya.onnx");
  const url = `https://huggingface.co/${repo}/resolve/main/laya.onnx`;
  process.stderr.write(
    `sentrygate: fetching the guard model (~440 MB, first run only) from ${repo} ...\n`,
  );
  try {
    mkdirSync(dest, { recursive: true });
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    await pipeline(Readable.fromWeb(res.body as any), createWriteStream(out));
    return out;
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw new Error(
      `sentrygate could not download the model from ${repo} (${msg}).\n` +
        "Pull it by hand and sentrygate picks it up on its own:\n" +
        `  hf download ${repo} laya.onnx --local-dir ./models\n` +
        '  export SENTRYGATE_ONNX="$PWD/models/laya.onnx"',
    );
  }
}

/**
 * Resolve the tokenizer directory and ONNX path, downloading the weights on
 * first use when nothing points at a local file.
 */
export async function resolveModel(
  modelDir?: string,
  onnxPath?: string,
): Promise<{ modelDir: string; onnxPath: string }> {
  let op = onnxPath ?? process.env.SENTRYGATE_ONNX;
  if (!op) {
    const local = join(process.cwd(), "models", "laya.onnx");
    op = existsSync(local) ? local : await fetchLayaOnnx();
  } else if (!existsSync(op)) {
    throw new Error(
      `sentrygate: could not find the ONNX graph at ${op}. Set SENTRYGATE_ONNX or pass { onnxPath }.`,
    );
  }
  const md = resolveTokenizerDir(modelDir, op);
  return { modelDir: md, onnxPath: op };
}
