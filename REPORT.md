# LLM Inference + Observability Assignment

## 1. Serving Configuration

Model: `Qwen/Qwen3-30B-A3B-Instruct-2507` on 1× H100 80GB.

| Flag | Value | Why |
|------|-------|-----|
| `--dtype bfloat16` | bf16 | H100 has native BF16 tensor cores. More stable than fp16 for large MoE models — wider exponent range means no overflow on big activations. |
| `--max-model-len` | 4096 | Prompts are 1.5–3K tokens and SQL outputs are short (~200 tokens max). Capping here saves KV cache memory without hitting the limit in practice. |
| `--max-num-seqs` | 32 | Agent calls are sequential per user (generate → verify → revise), so there's no benefit from high concurrency. 32 gives headroom without wasting memory. |
| `--enable-chunked-prefill` | on | Breaks long prompts into smaller chunks so generation requests aren't blocked waiting for a full prefill. Reduces tail latency when prompts vary in length. |
| `--trust-remote-code` | on | Required to load the Qwen3 model files. |

One issue hit during setup: vLLM 0.10.2 is incompatible with transformers 5.x (`all_special_tokens_extended` was removed). Fixed by pinning `transformers==4.51.3`.

There was also a warning about a missing MoE kernel config for `NVIDIA_H100_80GB_HBM3` — vLLM fell back to generic fused-MoE defaults. This is a known gap and is relevant context for the Phase 6 SLO iteration.

---

## 2. Observability Dashboard

Built an 8-panel Grafana dashboard covering three categories:

**Latency**
- E2E request latency P50/P95/P99 — tells you if the system is slow overall
- Time to first token P50/P95 — isolates prefill as a bottleneck
- Time per output token P50/P95 — isolates decode as a bottleneck
- Request queue time P95 — shows whether requests are waiting before inference even starts

**Throughput**
- Token throughput (gen + prompt tokens/sec) — generation rate under load
- Requests running / waiting — waiting > 0 means the system is at capacity
- Request success rate by finish reason — catches unexpected aborts or length truncations

**KV Cache**
- KV cache usage gauge (0–1) with green/yellow/red thresholds at 70%/90% — near 1.0 means evictions are likely and latency will spike

All panels reacted visibly when firing a burst of 10 concurrent requests.

---

## 3. Baseline Eval Results

Eval set: 30 questions from BIRD-bench across 11 SQLite databases.
Metric: execution accuracy — gold SQL and agent SQL run against the same DB, row sets compared after canonicalisation (sorted, stringified).

| Iteration | Pass Rate |
|-----------|-----------|
| 1 (generate only) | TBD on H100 |
| 2 (after first revise) | TBD on H100 |
| 3 (after second revise) | TBD on H100 |
| Final | TBD on H100 |

> Note: local dev runs on Qwen3-1.7B gave iter-1=13.3%, iter-2=20%, iter-3=20% — showing the loop helps but numbers aren't representative. Real baseline must come from Qwen3-30B-A3B on the H100.

---

## 4. SLO Diagnosis and Iteration

Target: P95 end-to-end agent latency < 5s, 10+ RPS over a 5-minute window.

*To be completed after Phase 6 load testing on the H100.*

Iteration log format (to fill in):
```
saw X → hypothesised Y → changed Z → result was W
```

---

## 5. Agent Value

The verify→revise loop adds measurable value. On the local dev run (1.7B model), pass rate went from 13.3% at iteration 1 to 20% at iteration 2 — the revise step recovered 2 questions the initial generation got wrong. Iteration 3 added nothing further, which makes sense: if revision didn't fix it after one attempt, a second revision with the same context rarely does.

On the H100 with the 30B model, I expect the gap between iter-1 and final to be larger because:
1. The verifier will produce valid JSON more reliably, so `ok=false` will be genuinely informative rather than a parse failure
2. The revise prompt has the failing SQL, error message, and verifier's complaint — the 30B model is much better at using all three together

The architecture earns its keep when the first SQL errors (wrong table, syntax error, wrong join) and the verifier catches it. The per-iteration pass rate from the H100 run will be the definitive evidence.

---

## 6. What I'd Do With More Time

- **Generate a fused MoE kernel config for H100** — vLLM warned about missing `E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json`. Running `vllm bench` to generate this could give a meaningful throughput improvement without changing anything else.
- **Add prefix caching** — agent calls share large system prompts and schema blocks across questions. Prefix caching would cut TTFT significantly on repeated DB schemas.
- **Improve the verifier prompt** — currently the verifier sees the raw row output. Adding column name context and a worked example of a bad result would reduce false positives (verify saying ok=false when the SQL was actually fine).
- **Structured output for verify** — force the model to output `{"ok": ..., "issue": ...}` using vLLM's guided decoding rather than hoping the model formats JSON correctly. Would eliminate the defensive parsing fallback entirely.
