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
| 1 (generate only) | 33.3% |
| 2 (after first revise) | 33.3% |
| 3 (after second revise) | 33.3% |
| Final | 33.3% (10/30) |

The loop added no value on the 30B model — all iterations came out the same. The verifier was passing results too easily, so the revise step never triggered on genuinely wrong answers. The verify prompt needs to be more strict to catch cases where the SQL returns the wrong columns or a suspiciously round number.

Post-tuning eval (after the schema fix): 30% (9/30). The fix slightly hurt quality on one question — the european_football_2 schema now renders `"unknown"` for NULL FK columns which confused the model on one edge case. Acceptable tradeoff given it eliminated all 500 errors.

---

## 4. SLO Diagnosis and Iteration

Target: P95 end-to-end agent latency < 5s, 10+ RPS over a 5-minute window.

**Baseline (2 RPS):** P95=6.04s, 30 HTTP 500 errors, achieved 1.33 RPS.

**Iteration 1:** saw HTTP 500s on ~12% of requests → hypothesised schema rendering crash on the european_football_2 database which has NULL FK column names → fixed `_q()` in `agent/schema.py` to handle None gracefully → P95 dropped to 4.56s, 0 errors. P95 now under 5s SLO at 2 RPS. ✅

**Iteration 2:** pushed to 10 RPS over 5 minutes → 1079 timeouts, P95=104s, only 43% of requests succeeded → SLO missed. The root cause is that agent runs make 1-2 sequential LLM calls, each ~1s. At 10 concurrent RPS that's 10-20 simultaneous vLLM requests stacking up faster than they drain. The system can sustain ~2 RPS comfortably but falls apart at 10.

**Verdict:** SLO partially hit — P95 < 5s achieved at 2 RPS after fixing the 500 errors. 10 RPS target missed. Gap at 10 RPS: P95 104s vs 5s target.

---

## 5. Agent Value

On the H100 with Qwen3-30B-A3B, the verify→revise loop didn't add measurable value — pass rate was flat at 33.3% across all iterations. This is different from what the local 1.7B dev runs suggested (13.3% → 20% improvement), and the reason is clear in hindsight: the verifier on the 30B model was too lenient. It kept returning `ok=true` even on wrong answers, so revise never fired on questions that actually needed fixing.

The architecture has the right shape — the loop should help — but the verifier prompt needs to be more aggressive about catching wrong answers. With a stricter verify prompt I'd expect the loop to start earning its keep. The per-iteration pass rate being flat is the evidence that the bottleneck is in the verifier, not the reviser.

---

## 6. What I'd Do With More Time

- **Stricter verifier prompt** — the flat per-iteration pass rate shows the verifier is the weak link. Adding few-shot examples of bad results (wrong columns, off-by-one counts, NULL when a value is expected) would make `ok=false` fire on genuine failures rather than just errors and zero rows.
- **Structured output for verify** — force the model to output `{"ok": ..., "issue": ...}` using vLLM's guided decoding rather than hoping the model formats JSON correctly. Would eliminate the defensive parsing fallback entirely.
- **Generate a fused MoE kernel config for H100** — vLLM warned about missing `E=128,N=768,device_name=NVIDIA_H100_80GB_HBM3.json`. Running `vllm bench` to generate this could give a meaningful throughput improvement at 10 RPS without changing anything else.
- **Prefix caching** — agent calls share large system prompts and schema blocks across questions from the same DB. Prefix caching would cut TTFT significantly on repeated schemas, which is exactly the pattern in a real analytics product where the same warehouse is queried repeatedly.
