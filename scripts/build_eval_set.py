#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the multi-slice eval corpus the MER gate needs (docs/EVAL.md, docs/RESEARCH.md).

The gating "orthographic-vs-acoustic split test" is unrunnable until paired audio+ref exist for each
slice. This pulls ≥N utts/slice via HuggingFace **streaming** (so it fetches only the samples it takes,
not the whole multi-GB dataset), resamples to 16 kHz mono, and writes:

    data/audio/<slice>/<id>.wav          16 kHz mono clips
    data/text/<slice>.ref.tsv            id<TAB>reference text  (for scripts/eval_asr.py)
    data/eval/manifest.tsv               slice<TAB>id<TAB>wav<TAB>text  (combined index)

Slices (see docs/DATASETS.md for accent/license):
    tw_cs    NTUML2021  (Taiwan zh-en code-switch, MIT)        <- the slice that matters most
    cs_eval  CSZS-zh-en (zh-en CS eval, MIT)
    ascend   ASCEND     (mixed-accent zh-en CS, CC-BY-SA)
    en       LibriSpeech test-clean (English retention, CC-BY)
    zh_cn    AISHELL-1 test (mainland-zh retention, Apache-2.0)
Common Voice zh-TW is gated behind Mozilla Data Collective (CC0) — fetch manually, then
`--add-manual zh_tw <dir>` to fold it in.

Usage:
    python scripts/build_eval_set.py --dry-run                      # show the plan, no deps/network
    pip install "datasets>=2.18" soundfile librosa
    python scripts/build_eval_set.py --slices tw_cs cs_eval --per-slice 80
    python scripts/build_eval_set.py --slices tw_cs --to-traditional   # refs -> Traditional (match s2twp hyp)
"""
import argparse
import os
import sys

# repo, config, split, audio col, candidate text cols (first present non-empty wins), label
SLICES = {
    "tw_cs":   ("ky552/ML2021_ASR_ST", None, "test", "audio",
                ["text", "sentence", "transcription", "chinese", "zh", "transcript"], "Taiwan zh-en CS (NTUML2021, MIT)"),
    "cs_eval": ("ky552/cszs_zh_en", None, "test", "audio",
                ["text", "sentence", "transcription", "transcript"], "zh-en CS eval (CSZS, MIT)"),
    "ascend":  ("CAiRE/ASCEND", None, "test", "audio",
                ["transcription", "text", "sentence"], "mixed-accent zh-en CS (ASCEND, CC-BY-SA)"),
    "en":      ("openslr/librispeech_asr", "clean", "test", "audio",
                ["text", "sentence"], "English retention (LibriSpeech, CC-BY)"),
    "zh_cn":   ("AISHELL/AISHELL-1", None, "test", "audio",
                ["transcription", "text", "sentence"], "mainland-zh retention (AISHELL-1, Apache-2.0)"),
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_text(example, candidates):
    for k in candidates:
        if k in example and isinstance(example[k], str) and example[k].strip():
            return example[k].strip()
    # last resort: any non-empty string field that isn't a path/id
    for k, v in example.items():
        if isinstance(v, str) and v.strip() and k not in ("id", "path", "file", "client_id"):
            return v.strip()
    return None


def fetch_slice(name, per_slice, to_traditional):
    from datasets import load_dataset, Audio  # lazy
    import soundfile as sf

    repo, config, split, audio_key, text_cands, label = SLICES[name]
    print(f"[{name}] streaming {repo}" + (f"/{config}" if config else "") + f" split={split}  ({label})")
    ds = load_dataset(repo, config, split=split, streaming=True)
    ds = ds.cast_column(audio_key, Audio(sampling_rate=16000))

    post = None
    if to_traditional:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from zh_tw_postproc import ZhTwPostProcessor
        post = ZhTwPostProcessor("s2twp")

    adir = os.path.join(ROOT, "data", "audio", name)
    os.makedirs(adir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "text"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "data", "eval"), exist_ok=True)
    ref_path = os.path.join(ROOT, "data", "text", f"{name}.ref.tsv")
    rows = []
    n = 0
    with open(ref_path, "w", encoding="utf-8") as ref:
        for i, ex in enumerate(ds):
            if n >= per_slice:
                break
            text = pick_text(ex, text_cands)
            au = ex.get(audio_key)
            if not text or not au or "array" not in au:
                continue
            uid = f"{name}_{n:04d}"
            wav = os.path.join(adir, f"{uid}.wav")
            sf.write(wav, au["array"], 16000)
            if post is not None:
                text = post(text)
            ref.write(f"{uid}\t{text}\n")
            rows.append((name, uid, os.path.relpath(wav, ROOT), text))
            n += 1
    print(f"[{name}] wrote {n} clips -> data/audio/{name}/  + {os.path.relpath(ref_path, ROOT)}")
    if n < per_slice:
        print(f"[{name}] WARNING: only {n}/{per_slice} usable utts (text/audio missing for some rows)",
              file=sys.stderr)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slices", nargs="+", default=["tw_cs", "cs_eval", "en", "zh_cn"],
                    help=f"subset of {list(SLICES)}")
    ap.add_argument("--per-slice", type=int, default=80, help="utts per slice (>=70 recommended)")
    ap.add_argument("--to-traditional", action="store_true",
                    help="run s2twp on CJK refs so they match post-s2twp hyp (use for Simplified-ref sources)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; no deps, no network")
    args = ap.parse_args()

    unknown = [s for s in args.slices if s not in SLICES]
    if unknown:
        ap.error(f"unknown slices {unknown}; choose from {list(SLICES)}")

    if args.dry_run:
        print(f"Plan: {args.per_slice} utts/slice into data/audio/<slice>/ + data/text/<slice>.ref.tsv")
        for s in args.slices:
            repo, config, split, _, _, label = SLICES[s]
            print(f"  {s:8s} <- {repo}{('/' + config) if config else ''} [{split}]   {label}")
        print("\nThen: python scripts/eval_asr.py --hyp <decoded>.tsv --ref data/text/<slice>.ref.tsv")
        print("Install first: pip install \"datasets>=2.18\" soundfile librosa")
        return

    all_rows = []
    for s in args.slices:
        try:
            all_rows += fetch_slice(s, args.per_slice, args.to_traditional)
        except Exception as e:  # noqa: BLE001 - one bad slice shouldn't kill the rest
            print(f"[{s}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    man = os.path.join(ROOT, "data", "eval", "manifest.tsv")
    if all_rows:
        os.makedirs(os.path.dirname(man), exist_ok=True)
        with open(man, "w", encoding="utf-8") as f:
            for r in all_rows:
                f.write("\t".join(r) + "\n")
        print(f"\nwrote {len(all_rows)} rows -> {os.path.relpath(man, ROOT)}")


if __name__ == "__main__":
    main()
    # numba/scipy can segfault in atexit finalizers on this platform AFTER all artifacts are
    # written; flush and hard-exit to avoid a spurious post-success crash/core dump.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
