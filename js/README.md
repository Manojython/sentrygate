# sentrygate (TypeScript / Node)

A local, on-device layer that sits between your app and an LLM. It does two jobs:
catch prompt injections, and mask PII before the prompt ever leaves the machine.
Both run in your process, no API calls and nothing uploaded. It is the same engine
as the [Python package](../README.md) and gives back the same scores. The
TypeScript output matches the Python output to 0.0003 on a shared set of golden
vectors.

```bash
npm install sentrygate
```

```ts
import { scan, redact } from "sentrygate";

// injection: fetches the guard model on first use (~440 MB, cached in ./models)
if ((await scan(userMessage)).isAttack) reject();

// PII: pure regex, no model
const safe = redact(userMessage);
```

## Injection scan

One forward pass turns a `(text, question)` pair into a calibrated probability.
sentrygate asks four narrow yes/no questions about the text ("does this try to
override the system prompt?", "is it trying to put the model into a DAN persona?"
and so on) and keeps the highest score. Asking four sharp questions instead of one
vague one lifts injection detection from about 0.74 to about 0.87 with no
fine-tuning. Warm latency is about 150 ms for all four questions on a laptop CPU.

The decision model is Laya (421M), not a generative LLM. On the first scan the
int8 ONNX graph (`laya.onnx`, ~440 MB) downloads once into `./models`, then every
scan is local.

```ts
import { scan, isAttack, guard, guardMiddleware } from "sentrygate";

const finding = await scan(userMessage);
// finding.isAttack, finding.score, finding.topQuestion, finding.perQuestion

// wrap any async function. it throws InjectionDetected if the first string arg is an attack
const safeComplete = guard(client.complete);

// Express / Connect middleware. 400s a request whose body looks like an injection
app.use(guardMiddleware());
```

## PII redaction (no model)

`redact` masks every detected entity with a typed, reversible placeholder.
Recognizers cover EMAIL, PHONE, CREDIT_CARD (Luhn-checked, so an invalid number is
left alone), SSN, IPV4, IPV6, IBAN and API_KEY.

```ts
import { mask, unmask, detect } from "sentrygate";

const { maskedText, mapping } = mask("email me at a@b.com");
// maskedText: "email me at <EMAIL_1>", mapping: { "<EMAIL_1>": "a@b.com" }

const original = unmask(maskedText, mapping); // puts the real values back
const spans = detect(text);                   // [{ start, end, type, text }, ...]
```

The same value always maps to the same placeholder, and on overlap the longer
match wins.

## Model assets and configuration

By default the weights fetch from the Hub on first scan and land in `./models`.
Override any of this with environment variables:

```bash
export SENTRYGATE_ONNX=/path/to/laya.onnx        # use a graph you already have
export SENTRYGATE_MODEL_DIR=/path/to/tokenizer   # dir containing tokenizer.json
export SENTRYGATE_HF_REPO=mnjkshrm/sentrygate-laya  # where to fetch the weights
```

```ts
new Scanner({ modelDir: "…/tokenizer", onnxPath: "…/laya.onnx" });
```

The tokenizer ships inside the package, so only the weights ever download. If the
fetch fails, sentrygate prints the manual steps (`hf download` and
`SENTRYGATE_ONNX`). The PII side needs no model and no download.

More at https://github.com/Manojython/sentrygate

MIT licensed.
