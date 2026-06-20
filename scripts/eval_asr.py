#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MER / CER / WER + code-switch boundary metrics for zh-TW/en hyp vs ref.

zh-en code-switch can't be scored by a single WER or CER (docs/EVAL.md). MER = CER on CJK characters +
WER on English words, computed over the whole utterance with a unified token stream:
  - CJK characters are each one token,
  - maximal runs of [A-Za-z0-9'] are one (lowercased) token,
  - everything else (punctuation/space) is dropped before alignment.
This yields one edit-distance whose denominator is (CJK chars + English words); per-language slices are
reported too. Boundary-F1 scores whether en<->zh switch points land where the reference has them.

Input: two TSV files, `id<TAB>text`, joined on id.

Usage:
    python scripts/eval_asr.py --hyp hyp.tsv --ref ref.tsv --metric mer
    python scripts/eval_asr.py --hyp hyp.tsv --ref ref.tsv --metric boundary
"""
import argparse
import re
import sys

_CJK = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_WORD = re.compile(r"[A-Za-z0-9']+")


def tokenize(text):
    """-> list of (token, lang) where lang in {'zh','en'}; punctuation/space dropped."""
    toks = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if _CJK.match(ch):
            toks.append((ch, "zh"))
            i += 1
        elif ch.isalnum() or ch == "'":
            m = _WORD.match(text, i)
            toks.append((m.group(0).lower(), "en"))
            i = m.end()
        else:
            i += 1
    return toks


def levenshtein(a, b):
    """Edit distance over token lists; returns (dist, substitutions, insertions, deletions)."""
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[la][lb]


def load(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            uid, _, text = line.partition("\t")
            d[uid] = text
    return d


def boundaries(toks):
    """Set of positions (token index) where language switches relative to previous kept token."""
    bset = set()
    prev = None
    for idx, (_, lang) in enumerate(toks):
        if prev is not None and lang != prev:
            bset.add(idx)
        prev = lang
    return bset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--metric", default="mer", choices=["mer", "boundary"])
    args = ap.parse_args()

    hyp, ref = load(args.hyp), load(args.ref)
    ids = [u for u in ref if u in hyp]
    missing = [u for u in ref if u not in hyp]
    if missing:
        print(f"[eval] WARNING: {len(missing)} ref ids missing from hyp", file=sys.stderr)
    if not ids:
        raise SystemExit("[eval] no overlapping ids between hyp and ref")

    if args.metric == "mer":
        tot_err = tot_ref = 0
        zh_err = zh_ref = en_err = en_ref = 0
        for u in ids:
            h, r = tokenize(hyp[u]), tokenize(ref[u])
            tot_err += levenshtein([t for t, _ in h], [t for t, _ in r])
            tot_ref += len(r)
            hz, rz = [t for t, l in h if l == "zh"], [t for t, l in r if l == "zh"]
            he, re_ = [t for t, l in h if l == "en"], [t for t, l in r if l == "en"]
            zh_err += levenshtein(hz, rz); zh_ref += len(rz)
            en_err += levenshtein(he, re_); en_ref += len(re_)
        mer = tot_err / max(tot_ref, 1)
        cer = zh_err / max(zh_ref, 1)
        wer = en_err / max(en_ref, 1)
        print(f"utts={len(ids)}")
        print(f"MER  = {mer:.4f}  ({tot_err}/{tot_ref})")
        print(f"zhCER= {cer:.4f}  ({zh_err}/{zh_ref})")
        print(f"enWER= {wer:.4f}  ({en_err}/{en_ref})")
    else:
        tp = fp = fn = 0
        for u in ids:
            hb, rb = boundaries(tokenize(hyp[u])), boundaries(tokenize(ref[u]))
            tp += len(hb & rb); fp += len(hb - rb); fn += len(rb - hb)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"utts={len(ids)}  switch-points: tp={tp} fp={fp} fn={fn}")
        print(f"boundary  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")


if __name__ == "__main__":
    main()
