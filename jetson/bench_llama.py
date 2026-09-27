"""Run llama-bench tests on the Jetson while sampling telemetry; print one JSON result.

Each test runs as its own llama-bench process so its telemetry window is clean.
Only the timed repetitions (end of the run, length = sum of llama-bench samples)
are used for the telemetry averages, not model load or warmup.

Test spec: pp<N>[@d<depth>] or tg<N>[@d<depth>], e.g. pp512 tg128 tg128@d4096
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from telemetry import Telemetry, summarize

HERE = os.path.dirname(os.path.abspath(__file__))
LLAMA_BENCH = os.path.expanduser("~/llama.cpp/build/bin/llama-bench")
MEMBW_RESULT = os.path.join(HERE, "membw.json")
THEORETICAL_BW_GBS = 3199e6 * 32 / 1e9  # 102.4 GB/s, see telemetry.BYTES_PER_EMC_CLOCK
TEARDOWN_S = 0.3  # ignore the last samples while llama-bench frees the model


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip()


def system_info():
    return {
        "hostname": socket.gethostname(),
        "power_mode": sh("nvpmodel -q 2>/dev/null | head -1").removeprefix("NV Power Mode: "),
        "l4t": sh("dpkg-query -W -f='${Version}' nvidia-l4t-core 2>/dev/null"),
        "jetpack": sh("dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null"),
        "cuda": sh("/usr/local/cuda/bin/nvcc --version | grep -oE 'release [0-9.]+'").removeprefix("release "),
        "kernel": sh("uname -r"),
    }


def parse_spec(spec):
    m = re.fullmatch(r"(pp|tg)(\d+)(?:@d(\d+))?", spec)
    if not m:
        sys.exit(f"bad test spec: {spec}")
    return m[1], int(m[2]), int(m[3] or 0)


def run_test(tel, model, spec, reps, extra):
    kind, n, depth = parse_spec(spec)
    cmd = [LLAMA_BENCH, "-m", model, "-ngl", "99", "-r", str(reps), "-o", "json",
           "-p", str(n if kind == "pp" else 0), "-n", str(n if kind == "tg" else 0),
           "-d", str(depth), *extra]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    if r.returncode != 0:
        sys.exit(f"llama-bench failed for {spec}:\n{r.stderr[-2000:]}")
    res = json.loads(r.stdout)[0]

    active_s = sum(res["samples_ns"]) / 1e9
    win_end = t1 - TEARDOWN_S
    tele = summarize(tel.window(win_end - active_s, win_end))

    tps = res["avg_ts"]
    derived = {}
    if "power_vdd_in_w_avg" in tele:
        derived["j_per_token"] = round(tele["power_vdd_in_w_avg"] / tps, 4)
    if "mem_bw_gbs_avg" in tele:
        bw = tele["mem_bw_gbs_avg"]
        derived["mem_bw_pct_theoretical"] = round(100 * bw / THEORETICAL_BW_GBS, 1)
        if os.path.exists(MEMBW_RESULT):
            achievable = json.load(open(MEMBW_RESULT))["read_gbs"]
            derived["mem_bw_pct_achievable_read"] = round(100 * bw / achievable, 1)
    if tele["n_samples"] < 10:
        derived["warning"] = "few telemetry samples; increase repetitions"

    return {
        "test": spec, "kind": kind, "n_tokens": n, "depth": depth, "reps": reps,
        "tokens_per_s": round(tps, 2), "tokens_per_s_stddev": round(res["stddev_ts"], 2),
        "wall_s": round(t1 - t0, 2), "active_s": round(active_s, 2),
        "telemetry": tele, "derived": derived,
        "llama": {k: res.get(k) for k in ("build_commit", "model_type", "model_size", "model_n_params",
                                           "n_batch", "n_ubatch", "flash_attn", "type_k", "type_v")},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tests", nargs="+", default=["pp512", "tg128"])
    ap.add_argument("--reps-pp", type=int, default=20)
    ap.add_argument("--reps-tg", type=int, default=3)
    ap.add_argument("--idle-s", type=float, default=3, help="idle baseline before the tests")
    ap.add_argument("--cooldown-s", type=float, default=5, help="pause between tests")
    ap.add_argument("--label", default="")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []  # passed through to llama-bench
    a = ap.parse_args(argv[:len(argv) - len(extra) - (1 if "--" in argv else 0)])
    model = os.path.expanduser(a.model)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": a.label,
        "model_file": os.path.basename(model),
        "model_bytes": os.path.getsize(model),
        "llama_bench_extra_args": extra,
        "system": system_info(),
        "membw": json.load(open(MEMBW_RESULT)) if os.path.exists(MEMBW_RESULT) else None,
        "tests": [],
    }

    with Telemetry() as tel:
        t = time.time()
        time.sleep(a.idle_s)
        result["idle"] = summarize(tel.window(t, time.time()))
        for i, spec in enumerate(a.tests):
            if i:
                time.sleep(a.cooldown_s)
            reps = a.reps_pp if spec.startswith("pp") else a.reps_tg
            print(f"running {spec} ...", file=sys.stderr, flush=True)
            result["tests"].append(run_test(tel, model, spec, reps, extra))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
