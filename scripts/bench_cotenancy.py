#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Co-tenancy benchmark: measure STT under TTS contention — the REAL 2-core gate.

Every RTF number in this repo (the ~0.39 budget) is STT measured **solo**. In the attendant, STT shares
the Tegra X1 with TTS: even pinned to disjoint cores they contend for memory bandwidth and L2 cache. This
harness runs the STT benchmark twice — alone, then while a TTS-like load saturates the other cores — and
reports the slowdown. That delta, not the solo figure, is the number the budget must clear.

How it works:
  1. Run `--stt-cmd` pinned (taskset) to `--stt-cores`, idle background  -> baseline.
  2. Spawn CPU-load workers pinned to `--tts-cores` (or run the real `--tts-cmd`), then run `--stt-cmd`
     again on `--stt-cores`  -> contended.
  3. Report wall-time and (if the stt-cmd prints an `RTF=` line, e.g. bench_nano.py) the RTF delta.

On the Nano: `--stt-cores 0,1 --tts-cores 2,3` measures the realistic shared-bandwidth contention. The
synthetic load approximates TTS's draw; pass `--tts-cmd "<launch jetson-tts>"` for true fidelity.

Usage:
    # real: STT on cores 0-1, TTS-like load on 2-3
    python scripts/bench_cotenancy.py --stt-cores 0,1 --tts-cores 2,3 \
        --stt-cmd "python scripts/bench_nano.py --model-dir <m> --int8 --threads 2 --wav <wav>"
    # prove the harness measures contention (no sherpa needed):
    python scripts/bench_cotenancy.py --selftest
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from multiprocessing import Process, Event


def _have_taskset():
    return subprocess.call(["bash", "-lc", "command -v taskset >/dev/null"]) == 0


def _wrap(cores, cmd):
    """Prefix a shell command with taskset for the given core list (or run as-is if unavailable)."""
    if cores and _have_taskset():
        return f"taskset -c {cores} {cmd}"
    return cmd


def _load_worker(stop, mat):
    """CPU-bound: repeated matmul to keep a core busy, emulating the TTS engine's draw."""
    import numpy as np
    a = np.random.rand(mat, mat).astype("float32")
    b = np.random.rand(mat, mat).astype("float32")
    while not stop.is_set():
        a = (a @ b) / mat  # divide by mat keeps entries ~O(1) (no overflow); pure FLOP burn


def start_load(cores, n_workers, mat=384):
    """Start n_workers CPU-load processes pinned (best-effort) to `cores`."""
    stop = Event()
    procs = []
    for _ in range(n_workers):
        p = Process(target=_load_worker, args=(stop, mat))
        p.start()
        if cores and _have_taskset():
            subprocess.call(["taskset", "-cp", cores, str(p.pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
    return stop, procs


def stop_load(stop, procs):
    stop.set()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()


def run_cmd(cores, cmd):
    full = _wrap(cores, cmd)
    t0 = time.perf_counter()
    r = subprocess.run(["bash", "-lc", full], capture_output=True, text=True)
    wall = time.perf_counter() - t0
    return wall, r.stdout + r.stderr


def parse_rtf(text):
    """Pull the best (min) RTF= value a bench tool printed, if any."""
    vals = [float(x) for x in re.findall(r"RTF[=\s]+([0-9.]+)", text)]
    # bench_nano prints a table; also catch 'rtf' column rows like ' 2  0.391 ...'
    if not vals:
        vals = [float(x) for x in re.findall(r"^\s*\d+\s+([01]\.\d{3})\b", text, re.M)]
    return min(vals) if vals else None


def report(label, wall, out):
    rtf = parse_rtf(out)
    extra = f"  RTF={rtf:.3f}" if rtf is not None else ""
    print(f"  {label:11s} wall={wall:6.2f}s{extra}")
    return rtf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stt-cmd", help="command to benchmark (e.g. a scripts/bench_nano.py invocation)")
    ap.add_argument("--stt-cores", default="0,1", help="cores for STT (default 0,1)")
    ap.add_argument("--tts-cores", default="2,3", help="cores for the TTS-like load (default 2,3)")
    ap.add_argument("--tts-cmd", help="real co-tenant command to run during the contended pass "
                                      "(default: synthetic matmul load)")
    ap.add_argument("--tts-workers", type=int, default=2, help="synthetic load workers (if no --tts-cmd)")
    ap.add_argument("--selftest", action="store_true", help="prove the harness measures contention (no sherpa)")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())
    if not args.stt_cmd:
        ap.error("--stt-cmd is required (or use --selftest)")

    if not _have_taskset():
        print("[bench_cotenancy] WARNING: taskset not found — core pinning disabled; "
              "numbers reflect global contention, not the 2-core split.", file=sys.stderr)

    print(f"STT cores={args.stt_cores}  TTS cores={args.tts_cores}")
    print("solo (TTS idle):")
    w_solo, o_solo = run_cmd(args.stt_cores, args.stt_cmd)
    rtf_solo = report("STT alone", w_solo, o_solo)

    print("contended (TTS busy):")
    if args.tts_cmd:
        tproc = subprocess.Popen(["bash", "-lc", _wrap(args.tts_cores, args.tts_cmd)],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        stop = procs = None
    else:
        tproc = None
        stop, procs = start_load(args.tts_cores, args.tts_workers)
        time.sleep(0.5)
    try:
        w_cont, o_cont = run_cmd(args.stt_cores, args.stt_cmd)
        rtf_cont = report("STT + TTS", w_cont, o_cont)
    finally:
        if tproc is not None:
            tproc.terminate()
        else:
            stop_load(stop, procs)

    print("verdict:")
    print(f"  wall slowdown    {w_cont / w_solo:5.2f}x  ({w_solo:.2f}s -> {w_cont:.2f}s)")
    if rtf_solo and rtf_cont:
        print(f"  RTF under load   {rtf_solo:.3f} -> {rtf_cont:.3f}  "
              f"({'OK, <1 real-time' if rtf_cont < 1 else 'OVER BUDGET (>=1)'})")
        print("  GATE: the contended RTF is the budget number — must stay < ~0.6 with headroom for jitter.")


# ---------------------------------------------------------------------------
def _cpu_proxy_seconds(work=1200):
    """Deterministic CPU-bound proxy for an STT decode (matmul); returns its own wall time."""
    import numpy as np
    t0 = time.perf_counter()
    a = np.random.rand(work, work).astype("float32")
    b = np.random.rand(work, work).astype("float32")
    for _ in range(6):
        a = (a @ b) / work
    # touch result so it isn't optimized away
    _ = float(a.sum())
    return time.perf_counter() - t0


def _selftest():
    """Run a CPU-proxy 'STT' alone, then while load saturates the SAME cores (overlap), to prove the
    harness detects contention. Real runs use DISJOINT cores to measure shared-bandwidth contention."""
    proxy = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
        "import sys; sys.path.insert(0,'scripts'); "
        "from bench_cotenancy import _cpu_proxy_seconds as f; "
        "t=f(); print('RTF=%.3f'%(t/ (1.0)))"  # treat proxy seconds as a pseudo-RTF for the parser
    )
    cores = "0"  # overlap STT and load on one core to force contention deterministically
    print("[selftest] proxy STT on core", cores, "+ load on the SAME core (forces contention)")
    w_solo, o_solo = run_cmd(cores, proxy)
    print(f"  solo       wall={w_solo:5.2f}s")
    stop, procs = start_load(cores, 3, mat=256)
    time.sleep(0.3)
    try:
        w_cont, o_cont = run_cmd(cores, proxy)
    finally:
        stop_load(stop, procs)
    print(f"  contended  wall={w_cont:5.2f}s   slowdown={w_cont / w_solo:.2f}x")
    ok = w_cont > w_solo * 1.15  # contention must be visibly measurable
    print(f"[selftest] {'PASS' if ok else 'FAIL'} (harness {'detects' if ok else 'did NOT detect'} contention)",
          file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    main()
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
