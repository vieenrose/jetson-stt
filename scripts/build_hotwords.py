#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier-2 contextual biasing: build a sherpa-onnx hotwords file from a CN->TW term list.

sherpa-onnx contextual hotwords bias the transducer toward chosen phrases at decode time (requires
`decoding_method="modified_beam_search"`). This turns the curated Taiwan term / entity list into the
hotwords file sherpa expects: one phrase per line as space-separated *modeling units*, optionally with a
trailing `:<boost>`.

IMPORTANT (verified — docs/RESEARCH.md): X-ASR emits **Simplified** Chinese, so hotwords MUST be in the
model's Simplified token space, NOT Traditional — a Traditional hotword like `軟體` will not tokenize to
the model's units and silently never biases. This script emits the **Simplified** form: it prefers the
`cn_term` column, else converts `tw_term` with OpenCC `t2s` (the live demo does the same `t2s` step
before biasing, then `s2twp` on the output). For a char/BPE zipformer the modeling unit is the token;
pass --tokens to verify every char is in the vocab and warn on OOV.

Input TSV columns:  tw_term   [cn_term]   [boost]
    軟體            软件        2.5
    滑鼠            鼠标
    台積電                      3.0

Usage:
    python scripts/build_hotwords.py --terms data/tw_terms/cn_tw_terms.tsv \\
        --tokens ref/tokens.txt --boost 2.0 --out data/tw_terms/hotwords.txt
"""
import argparse
import sys


def load_vocab(tokens_path):
    vocab = set()
    with open(tokens_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split()
            if parts:
                vocab.add(parts[0])
    return vocab


def get_t2s():
    """OpenCC Traditional->Simplified converter, or None if opencc is unavailable."""
    try:
        from opencc import OpenCC
        return OpenCC("t2s")
    except ImportError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--terms", required=True, help="TSV: tw_term [cn_term] [boost]")
    ap.add_argument("--tokens", help="ref/tokens.txt — if given, warn on OOV chars")
    ap.add_argument("--boost", type=float, default=2.0, help="default per-phrase boost")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vocab = load_vocab(args.tokens) if args.tokens else None
    t2s = get_t2s()
    written, oov, no_simp = 0, 0, 0
    with open(args.terms, encoding="utf-8") as f, open(args.out, "w", encoding="utf-8") as out:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            tw = cols[0].strip()
            if not tw:
                continue
            cn = cols[1].strip() if len(cols) >= 2 and cols[1].strip() else ""
            boost = args.boost
            if len(cols) >= 3 and cols[2].strip():
                try:
                    boost = float(cols[2].strip())
                except ValueError:
                    pass
            # Bias in the model's SIMPLIFIED token space: prefer the cn_term column;
            # else convert the tw_term via OpenCC t2s; else fall back to tw (with a warning).
            if cn:
                simp = cn
            elif t2s is not None:
                simp = t2s.convert(tw)
            else:
                simp = tw
                no_simp += 1
            if vocab is not None:
                missing = [c for c in simp if c.strip() and c not in vocab]
                if missing:
                    print(f"[build_hotwords] OOV in '{simp}' (from '{tw}'): {''.join(missing)} (won't bias)",
                          file=sys.stderr)
                    oov += 1
            # sherpa hotwords: tokens space-separated; for char models, space out CJK chars.
            spaced = " ".join(list(simp))
            out.write(f"{spaced} :{boost}\n")
            written += 1
    if no_simp:
        print(f"[build_hotwords] WARNING: opencc not installed and {no_simp} terms had no cn_term column — "
              f"those were emitted as-is (Traditional) and likely WON'T bias. `pip install "
              f"opencc-python-reimplemented` or fill the cn_term column.", file=sys.stderr)
    print(f"[build_hotwords] wrote {written} Simplified hotwords -> {args.out} ({oov} with OOV chars)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
