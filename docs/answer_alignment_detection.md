# Answer-start boundary detection feasibility

We need a "final answer starts here" marker in generations so we can realign
divergence curves to token 0 = start of the answer (not token 0 = first generated
token). Measured on `chaos-scaffold-long-*` 512-token runs (n=84 generations per
model: 2 sides × pair_ids across 7 perturbation categories).

## Marker hit-rates

| model              | family          | hit% | dominant marker       | median scaffold tokens |
|--------------------|-----------------|------|-----------------------|------------------------|
| qwen3.5-4b         | reasoning       | 99%  | `**Drafting` / `**Final Selection` | 155 |
| qwen3.5-9b         | reasoning       | 98%  | `**Drafting` / `**Final Selection` | 166 |
| smollm3-3b         | reasoning       | 27%  | `</think>`            | 2 (most outputs `<think></think>` empty) |
| deepseek-r1-7b     | reasoning       | 33%  | `</think>`            | 392 (thinks too long) |
| phi4-reasoning+    | reasoning       | 0%   | none                  | — (repetition loop, never closes `<think>`) |
| mistral7b-v03      | non-reasoning   | 0%   | —                     | n/a |
| olmo3-7b           | non-reasoning   | 0%   | —                     | n/a |
| granite33-8b       | non-reasoning   | 0%   | —                     | n/a |

**Non-reasoning false-positive rate is 0%.** The Qwen "Thinking Process" family
has an extremely reliable marker (the model always drafts a numbered list then
emits `**Final ...**` / `**Drafting ...**`). DeepSeek-R1 and SmolLM3 use
`</think>`, but DeepSeek's reasoning budget often exceeds 512 tokens, so only
~⅓ of generations actually reach the final answer — those pairs must be dropped.
Phi-4-reasoning+ has a pathological repetition-loop failure mode at temperature
0 that never reaches the final answer within the budget → 100% drop.

## Recommended strategy

Ordered marker cascade:

1. `</think>` (DeepSeek-R1, SmolLM3, Phi-4 if it closed)
2. `**Final Answer:**` / `**Answer:**` / `Final Answer:` / `####`
3. `**Final Selection**` / `**Final Response**` / `**Drafting` (Qwen "Thinking
   Process" scaffold — the last numbered step before the polished answer)
4. If none hit → return `None` (drop from aligned plot)

For non-reasoning models and `thinkoff` variants, skip detection entirely —
answer-start is always 0.

## Expected drops after alignment

| model            | usable pairs (both sides have boundary) |
|------------------|------------------------------------------|
| qwen35-4b        | ~99% |
| qwen35-9b        | ~95% |
| deepseek-r1-7b   | ~10% (both sides need to finish) |
| smollm3-3b       | ~20% |
| phi4-reasoning+  | 0% → drop model from aligned panel |

Feasible. Proceed with the pipeline; expect phi-4 to be absent from the
answer-aligned view, which is itself a finding worth surfacing.
