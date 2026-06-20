#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""On-device streaming benchmark for X-ASR (sherpa-onnx): RTF + encoder ms/chunk + first-partial latency.

The hard gate from docs/EVAL.md: every candidate must stay real-time at the attendant's 2-core budget.
This drives the streaming recognizer over a wav, feeding fixed-size chunks, and reports per-thread-count:
  - end-to-end RTF (compute time / audio duration)
  - mean steady-state ms per fed chunk (warm-up excluded)
  - first-non-empty-partial latency

Enforce the attendant's core ceiling externally:  taskset -c 0,1 python scripts/bench_nano.py ...

Usage:
    python scripts/bench_nano.py --model-dir /path/to/x-asr/960ms --int8 \\
        --threads 1 2 3 4 --wav data/audio/codeswitch_ref.wav --chunk-ms 100
"""
import argparse
import glob
import os
import sys
import time


def _find(model_dir, *patterns):
    for p in patterns:
        hits = sorted(glob.glob(os.path.join(model_dir, p)))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"none of {patterns} in {model_dir}")


def build_recognizer(model_dir, int8, num_threads, decoding="greedy_search", hotwords=None):
    import sherpa_onnx

    suffix = ".int8.onnx" if int8 else ".onnx"
    enc = _find(model_dir, f"encoder*{suffix}", "encoder*.onnx")
    dec = _find(model_dir, f"decoder*{suffix}", "decoder*.onnx")
    joi = _find(model_dir, f"joiner*{suffix}", "joiner*.onnx")
    tokens = _find(model_dir, "tokens.txt")
    kw = dict(
        tokens=tokens, encoder=enc, decoder=dec, joiner=joi,
        num_threads=num_threads, provider="cpu",
        decoding_method=decoding, sample_rate=16000, feature_dim=80,
    )
    if hotwords and decoding == "modified_beam_search":
        kw.update(hotwords_file=hotwords, hotwords_score=2.0)
    return sherpa_onnx.OnlineRecognizer.from_transducer(**kw)


def run_once(recognizer, samples, sr, chunk_samples):
    """Feed fixed chunks; return (compute_s, n_chunks, first_partial_s, text)."""
    stream = recognizer.create_stream()
    first_partial = None
    n_chunks = 0
    t0 = time.perf_counter()
    for start in range(0, len(samples), chunk_samples):
        chunk = samples[start:start + chunk_samples]
        stream.accept_waveform(sr, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        n_chunks += 1
        if first_partial is None and recognizer.get_result(stream).strip():
            first_partial = time.perf_counter() - t0
    import numpy as np
    tail = np.zeros(int(0.5 * sr), dtype="float32")
    stream.accept_waveform(sr, tail)
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    compute_s = time.perf_counter() - t0
    return compute_s, n_chunks, first_partial, recognizer.get_result(stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--threads", type=int, nargs="+", default=[2])
    ap.add_argument("--chunk-ms", type=int, default=100, help="audio fed per step (feed cadence)")
    ap.add_argument("--decoding", default="greedy_search",
                    choices=["greedy_search", "modified_beam_search"])
    ap.add_argument("--hotwords", help="hotwords file (modified_beam_search only)")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    samples, sr = sf.read(args.wav, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if sr != 16000:
        print(f"[bench] WARNING: wav is {sr} Hz, model expects 16000", file=sys.stderr)
    audio_s = len(samples) / sr
    chunk_samples = int(args.chunk_ms / 1000 * sr)

    print(f"# {os.path.basename(args.wav)}  {audio_s:.2f}s audio  int8={args.int8}  decoding={args.decoding}")
    print(f"{'threads':>7} {'RTF':>6} {'ms/chunk':>9} {'1st-partial':>12} {'text':>4}")
    for nt in args.threads:
        rec = build_recognizer(args.model_dir, args.int8, nt, args.decoding, args.hotwords)
        for _ in range(args.warmup):
            run_once(rec, samples, sr, chunk_samples)
        best = None
        for _ in range(args.runs):
            compute_s, n_chunks, fp, text = run_once(rec, samples, sr, chunk_samples)
            if best is None or compute_s < best[0]:
                best = (compute_s, n_chunks, fp, text)
        compute_s, n_chunks, fp, text = best
        rtf = compute_s / audio_s
        ms_chunk = compute_s / max(n_chunks, 1) * 1000
        fp_str = f"{fp*1000:.0f}ms" if fp else "n/a"
        print(f"{nt:>7} {rtf:>6.3f} {ms_chunk:>8.1f}  {fp_str:>12}   {text[:60]!r}")


if __name__ == "__main__":
    main()
