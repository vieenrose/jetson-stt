#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deployable streaming zh-TW/en recognizer: X-ASR int8 (sherpa-onnx) → optional hotwords → s2twp.

This is the reference integration for the attendant's STT loop AND the Tier-1/2 measurement tool. It
wraps sherpa-onnx `OnlineRecognizer` with the project's two zero-retrain levers:
  - Tier-1: OpenCC s2twp post-processing on finalized text (`--s2twp`) — maximal-CJK-run conversion.
  - Tier-2: contextual hotwords (`--hotwords` + `--decoding-method modified_beam_search`).

It decodes a dir of wavs (or one --wav), writes id<TAB>text, and reports RTF (per real audio) at the given
thread count — so you can compare greedy vs modified_beam_search against the 2-core budget.

Usage (on the Nano, moss env):
    python stream_asr.py --model-dir ~/xasr-bench/int8-960 --tokens ~/xasr-bench/tokens.txt \
        --wav-dir tw_cs --out hyp.tsv --threads 2 --decoding-method greedy_search --s2twp
    # Tier-2:
    python stream_asr.py ... --decoding-method modified_beam_search --hotwords hotwords.txt --hotwords-score 2.0
"""
import argparse, glob, os, sys, time
import numpy as np
import soundfile as sf
import sherpa_onnx


def build(model_dir, tokens, threads, method, hotwords, hw_score):
    def f(*pats):
        for p in pats:
            h = sorted(glob.glob(os.path.join(model_dir, p)))
            if h:
                return h[0]
        raise FileNotFoundError(pats)
    kw = dict(
        tokens=tokens, encoder=f("encoder*.int8.onnx", "encoder*.onnx"),
        decoder=f("decoder*.onnx"), joiner=f("joiner*.int8.onnx", "joiner*.onnx"),
        num_threads=threads, provider="cpu", decoding_method=method,
        sample_rate=16000, feature_dim=80,
    )
    if hotwords and method == "modified_beam_search":
        kw.update(hotwords_file=hotwords, hotwords_score=hw_score)
    return sherpa_onnx.OnlineRecognizer.from_transducer(**kw)


def get_text(rec, s):
    r = rec.get_result(s)
    return (r if isinstance(r, str) else getattr(r, "text", str(r))).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--wav-dir")
    ap.add_argument("--wav")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--decoding-method", default="greedy_search",
                    choices=["greedy_search", "modified_beam_search"])
    ap.add_argument("--hotwords", help="hotwords file (modified_beam_search only)")
    ap.add_argument("--hotwords-score", type=float, default=2.0)
    ap.add_argument("--tail-pad", type=float, default=2.0, help="seconds of silence to flush the chunk")
    ap.add_argument("--s2twp", action="store_true", help="apply OpenCC s2twp (Tier-1) to output")
    args = ap.parse_args()

    post = None
    if args.s2twp:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from zh_tw_postproc import ZhTwPostProcessor
        post = ZhTwPostProcessor("s2twp")

    rec = build(args.model_dir, args.tokens, args.threads, args.decoding_method,
                args.hotwords, args.hotwords_score)
    wavs = [args.wav] if args.wav else sorted(glob.glob(os.path.join(args.wav_dir, "*.wav")))
    n = 0; t0 = time.perf_counter(); audio_s = 0.0
    with open(args.out, "w", encoding="utf-8") as out:
        for w in wavs:
            uid = os.path.splitext(os.path.basename(w))[0]
            samples, sr = sf.read(w, dtype="float32")
            if samples.ndim > 1:
                samples = samples.mean(axis=1)
            audio_s += len(samples) / sr
            s = rec.create_stream()
            s.accept_waveform(sr, samples)
            s.accept_waveform(sr, np.zeros(int(args.tail_pad * sr), dtype="float32"))
            s.input_finished()
            while rec.is_ready(s):
                rec.decode_stream(s)
            text = get_text(rec, s)
            if post is not None:
                text = post(text)
            out.write(f"{uid}\t{text}\n")
            n += 1
    dt = time.perf_counter() - t0
    print(f"stream_asr: {n} wavs ({audio_s:.1f}s) in {dt:.1f}s  RTF={dt/max(audio_s,1):.3f} "
          f"@ {args.threads} thr / {args.decoding_method}"
          f"{' +hotwords' if (args.hotwords and args.decoding_method=='modified_beam_search') else ''} "
          f"-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
