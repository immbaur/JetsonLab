# LLM inference benchmarking — tooling plan

Goal: measure every KPI that matters for LLM inference on the Jetson Orin Nano Super,
reproducibly, so models, quantizations and settings can be compared over time.

## Architecture

```
Mac (repo)                                   Jetson (~/jetsonlab)
tools/bench.py  ── rsync jetson/ ───────────▶ jetson/*.py, membw.cu
                ── ssh: run benchmark ──────▶ bench_llama.py ─┬─ llama-bench (-o json)
                                                              └─ telemetry.py (sudo tegrastats, 200 ms)
results/*.json  ◀── JSON result on stdout ───
tools/report.py ──▶ results/REPORT.md
```

- All measurement runs **on the Jetson** (local clock, no network jitter in timings).
- The Mac only syncs code, triggers runs and stores results in `results/` (committed).
- Telemetry is sampled while the benchmark runs; only the steady-state window
  (the timed repetitions, not model load / warmup) is used for the averages.

## KPIs

| Group | KPI | Source | Phase |
|---|---|---|---|
| Speed | Prompt processing t/s, generation t/s (± stddev) | `llama-bench` | 1 |
| Speed | Speed vs. context depth (0 / 4k / 16k …) | `llama-bench -d` | 1 |
| Memory BW | Used GB/s, % of theoretical (102.4 GB/s), % of measured achievable | `tegrastats` EMC + `membw` | 1 |
| Memory BW | Achievable DRAM bandwidth (read / write / copy) | `membw.cu` | 1 |
| Resources | GPU util % / clock, RAM peak, CPU util | `tegrastats` | 1 |
| Energy | Avg power (VDD_IN, CPU+GPU+CV, SOC), **J/token** | `tegrastats` | 1 |
| Thermal | Max GPU / Tj temperature, throttling over long runs | `tegrastats` | 1 (soak: 2) |
| Latency | TTFT, inter-token latency p50/p95, end-to-end | streaming client vs `llama-server` | 2 |
| Load | Throughput with 1/2/4/8 parallel requests | client + `llama-server --parallel` | 2 |
| Startup | Model load time (cold / warm) | `llama-server` log | 2 |
| Quality | Perplexity per quantization | `llama-perplexity` | 3 |
| Quality | Task accuracy on own test set, reasoning token overhead | eval script | 3 |

## Formulas

- Bandwidth [GB/s] = EMC% / 100 × EMC clock [MHz] × 10⁶ × 32 B
  (LPDDR5, 128-bit bus, 2 transfers/clock → 32 B/clock; 3199 MHz → 102.4 GB/s peak)
- Energy per token [J] = avg VDD_IN [W] / tokens per second

## Phases

1. **Core harness** (now): telemetry, bandwidth test, `llama-bench` runner, JSON results, report.
2. Server metrics: TTFT / ITL / parallel load via `llama-server`, soak test for throttling.
3. Quality: perplexity and task evals across quantizations (Q4_K_M vs QAD-Q4_0 vs Q8_0 …).

## Jetson prerequisites

- llama.cpp built in `~/llama.cpp/build/bin`, models in `~/models`.
- `/etc/sudoers.d/10-tegrastats`: `<user> ALL=(root) NOPASSWD: /usr/bin/tegrastats`
  (EMC / memory-bandwidth fields are only reported to root).
