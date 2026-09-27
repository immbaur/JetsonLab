"""Sample Jetson telemetry by running `tegrastats` in the background.

EMC (memory controller) fields are only reported to root, so tegrastats is
started through `sudo -n` (see /etc/sudoers.d/10-tegrastats).
"""

import re
import subprocess
import threading
import time

# LPDDR5 on a 128-bit bus: 2 transfers per clock x 16 bytes = 32 bytes per EMC clock.
BYTES_PER_EMC_CLOCK = 32

_RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_RE_CPU = re.compile(r"CPU \[([^\]]*)\]")
_RE_EMC = re.compile(r"EMC_FREQ (\d+)%@(\d+)")
_RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%(?:@\[?(\d+))?")
_RE_TEMP = re.compile(r"(\w+)@(-?[\d.]+)C")
_RE_RAIL = re.compile(r"(VDD_\w+) (\d+)mW")


def parse_line(line):
    """Parse one tegrastats line into a flat dict of numbers."""
    s = {}
    if m := _RE_RAM.search(line):
        s["ram_used_mb"], s["ram_total_mb"] = int(m[1]), int(m[2])
    if m := _RE_CPU.search(line):
        loads = [int(c.split("%")[0]) for c in m[1].split(",") if "%" in c]
        if loads:
            s["cpu_util_pct"] = sum(loads) / len(loads)
    if m := _RE_EMC.search(line):
        s["emc_util_pct"], s["emc_mhz"] = int(m[1]), int(m[2])
        s["mem_bw_gbs"] = s["emc_util_pct"] / 100 * s["emc_mhz"] * 1e6 * BYTES_PER_EMC_CLOCK / 1e9
    if m := _RE_GR3D.search(line):
        s["gpu_util_pct"] = int(m[1])
        if m[2]:
            s["gpu_mhz"] = int(m[2])
    for name, value in _RE_TEMP.findall(line):
        s[f"temp_{name}_c"] = float(value)
    for rail, mw in _RE_RAIL.findall(line):
        s[f"power_{rail.lower()}_w"] = int(mw) / 1000
    return s


class Telemetry:
    """Context manager that collects timestamped tegrastats samples."""

    def __init__(self, interval_ms=200):
        self.interval_ms = interval_ms
        self.samples = []  # list of (unix_time, dict)
        self._proc = None
        self._thread = None

    def __enter__(self):
        self._proc = subprocess.Popen(
            ["sudo", "-n", "tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self):
        for line in self._proc.stdout:
            self.samples.append((time.time(), parse_line(line)))

    def __exit__(self, *exc):
        # tegrastats runs as root, so it has to be stopped through tegrastats itself.
        subprocess.run(["sudo", "-n", "tegrastats", "--stop"], stderr=subprocess.DEVNULL)
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        self._thread.join(timeout=2)

    def window(self, start, end):
        return [s for t, s in self.samples if start <= t <= end]


def summarize(samples):
    """Average / max of every numeric field across samples."""
    if not samples:
        return {"n_samples": 0}
    keys = sorted({k for s in samples for k in s})
    out = {"n_samples": len(samples)}
    for k in keys:
        vals = [s[k] for s in samples if k in s]
        out[f"{k}_avg"] = round(sum(vals) / len(vals), 3)
        out[f"{k}_max"] = round(max(vals), 3)
    return out


if __name__ == "__main__":
    import json
    import sys

    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5
    with Telemetry() as tel:
        time.sleep(seconds)
    print(json.dumps(summarize([s for _, s in tel.samples]), indent=2))
