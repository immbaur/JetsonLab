# JetsonLab

LLM inference experiments on a Jetson Orin Nano Super (JetPack 7.2.1, llama.cpp + CUDA).

## Benchmarking

Runs on the Jetson over SSH (`jetson-orin`), results land in `results/` and `results/REPORT.md`.
See [docs/tooling-plan.md](docs/tooling-plan.md) for KPIs, formulas and phases.

```bash
# achievable DRAM bandwidth (read / write / copy)
tools/bench.py membw

# llama-bench + telemetry (power, J/token, memory bandwidth, GPU, RAM, temperature)
tools/bench.py llama LFM2.5-2.6B-Q4_K_M.gguf --tests pp512 tg128 tg128@d4096 --label baseline

# extra llama-bench args after --
tools/bench.py llama LFM2.5-2.6B-Q4_K_M.gguf --label no-fa -- -fa off

# rebuild the report from all results
tools/report.py
```

| Path | Contents |
|---|---|
| `jetson/` | Code synced to and run on the Jetson (`telemetry.py`, `bench_llama.py`, `membw.cu`) |
| `tools/` | Mac-side runner and report generator |
| `results/` | Raw JSON per run and the generated `REPORT.md` |
