#!/usr/bin/env python3
"""Run benchmarks on the Jetson from the Mac and store results in results/.

  tools/bench.py sync                      copy jetson/ to the Jetson
  tools/bench.py membw                     measure achievable DRAM bandwidth
  tools/bench.py llama MODEL [opts] [-- llama-bench args]
                                           run llama-bench tests with telemetry

Examples:
  tools/bench.py llama LFM2.5-2.6B-Q4_K_M.gguf
  tools/bench.py llama LFM2.5-2.6B-Q4_K_M.gguf --tests pp512 tg128 tg128@d4096 --label ctx
"""

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HOST = "jetson-orin"
REMOTE_DIR = "~/jetsonlab"
REMOTE_MODELS = "~/models"
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def ssh(cmd, capture=True):
    r = subprocess.run(["ssh", HOST, cmd], text=True,
                       stdout=subprocess.PIPE if capture else None)
    if r.returncode != 0:
        sys.exit(f"remote command failed ({r.returncode}): {cmd}")
    return r.stdout


def sync():
    ssh(f"mkdir -p {REMOTE_DIR}")
    subprocess.run(["rsync", "-a", "--exclude", "__pycache__", "--exclude", "membw",
                    "--exclude", "membw.json", f"{ROOT}/jetson/", f"{HOST}:{REMOTE_DIR}/"], check=True)


def save(prefix, name, data):
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RESULTS / f"{stamp}_{prefix}{'_' + name if name else ''}.json"
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"saved {path.relative_to(ROOT)}")


def membw(a):
    sync()
    out = ssh(f"cd {REMOTE_DIR} && ( [ membw -nt membw.cu ] || /usr/local/cuda/bin/nvcc -O3 -arch=sm_87 "
              f"-o membw membw.cu ) && ./membw {a.mib} {a.iters} | tee membw.json")
    data = json.loads(out)
    print(json.dumps(data, indent=2))
    save("membw", "", data)


def llama(a):
    sync()
    model = a.model if "/" in a.model else f"{REMOTE_MODELS}/{a.model}"
    cmd = (f"cd {REMOTE_DIR} && python3 bench_llama.py --model {model} --tests {' '.join(a.tests)} "
           f"--reps-pp {a.reps_pp} --reps-tg {a.reps_tg} --label {shlex.quote(a.label)}")
    if a.extra:
        cmd += " -- " + " ".join(shlex.quote(x) for x in a.extra)
    data = json.loads(ssh(cmd))
    for t in data["tests"]:
        tel, d = t["telemetry"], t["derived"]
        print(f"{t['test']:>14}  {t['tokens_per_s']:>8.1f} t/s  "
              f"{tel.get('power_vdd_in_w_avg', 0):5.1f} W  {d.get('j_per_token', 0):.3f} J/tok  "
              f"{tel.get('mem_bw_gbs_avg', 0):5.1f} GB/s ({d.get('mem_bw_pct_theoretical', 0)}%)  "
              f"GPU {tel.get('gpu_util_pct_avg', 0):.0f}%  {tel.get('temp_gpu_c_max', 0):.1f} °C")
    name = Path(a.model).stem + (f"_{a.label}" if a.label else "")
    save("llama", name, data)
    subprocess.run([sys.executable, str(ROOT / "tools" / "report.py")], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    p = sub.add_parser("membw")
    p.add_argument("--mib", type=int, default=512)
    p.add_argument("--iters", type=int, default=20)
    p = sub.add_parser("llama")
    p.add_argument("model", help="file name in ~/models on the Jetson, or a path")
    p.add_argument("--tests", nargs="+", default=["pp512", "tg128"])
    p.add_argument("--reps-pp", type=int, default=20)
    p.add_argument("--reps-tg", type=int, default=3)
    p.add_argument("--label", default="")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    a = ap.parse_args(argv[:len(argv) - len(extra) - (1 if "--" in argv else 0)])
    a.extra = extra
    {"sync": lambda _: sync(), "membw": membw, "llama": llama}[a.cmd](a)


if __name__ == "__main__":
    main()
