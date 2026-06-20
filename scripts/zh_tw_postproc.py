#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier-1 zh-TW post-processing: Simplified -> Traditional (Taiwan phrasing) + CJK normalization.

X-ASR emits Simplified Chinese with mainland word choices. Taiwan reads Traditional with Taiwan phrasing,
which is OpenCC `s2twp` (phrase-aware), not the glyph-only `s2tw`. Conversion is scoped to CJK spans so
English BPE output is never touched. Costs no model change and no extra CPU core — this is the dominant
"looks wrong for Taiwan" fix (see docs/ZH_TW_PLAN.md, TRAINING.md Tier 1, docs/RESEARCH.md).

CRITICAL (verified — docs/RESEARCH.md): convert **maximal CJK runs**, NOT one character at a time.
OpenCC's Taiwan-phrase layer (软件->軟體, 内存->記憶體, 头发->頭髮) operates on multi-character phrases;
feeding it single chars silently disables that layer and collapses s2twp to glyph-only s2tw — making this
whole step a near-no-op. This module converts whole CJK runs, reproducing whole-string s2twp output while
keeping English/ASCII byte-identical.

Two practical rules for the streaming loop:
  - Apply on **finalized** segments only, never on unstable partials (avoids flicker as OpenCC re-decides).
  - OpenCC over-converts a few ambiguous surnames (e.g. 余->餘 in 余先生). Pass --protect a TSV of
    `simplified<TAB>desired` overrides applied AFTER OpenCC to fix proper nouns it mangles.

Usage:
    python scripts/zh_tw_postproc.py --in hyp.tsv --out hyp_zhtw.tsv --opencc s2twp
    python scripts/zh_tw_postproc.py --selftest          # assert the phrase layer is actually working
    # library: from zh_tw_postproc import ZhTwPostProcessor; p = ZhTwPostProcessor(); p(text)
"""
import argparse
import re
import sys

# CJK ranges / punctuation, mirroring the upstream sherpa streaming frontend
# (x-asr-rapidspeech/ref/sherpa_streaming_infer.py) so spacing matches the deployed runtime.
_CJK_RANGE = r"㐀-䶿一-鿿豈-﫿"
_CJK_PUNCT = re.escape("，。！？；：、（）《》〈〉【】「」『』“”‘’")
_ASCII_PUNCT_NO_LEADING_SPACE = re.escape(",.!?;:%)]}")
# A *run* of one-or-more CJK chars — so OpenCC sees whole phrases, not isolated glyphs.
_CJK_RUN = re.compile(rf"[{_CJK_RANGE}]+")


def normalize_cjk_spacing(text: str) -> str:
    """Drop spurious spaces around CJK characters and punctuation."""
    text = re.sub(rf"(?<=[{_CJK_RANGE}])\s+(?=[{_CJK_RANGE}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_RANGE}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_RANGE}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"\s+(?=[{_ASCII_PUNCT_NO_LEADING_SPACE}])", "", text)
    return text


def load_protect(path):
    """TSV of `simplified<TAB>desired` post-OpenCC overrides for proper nouns OpenCC mangles."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) >= 2 and cols[0] and cols[1]:
                pairs.append((cols[0], cols[1]))
    return pairs


class ZhTwPostProcessor:
    """Callable: converts CJK runs to Taiwan Traditional, leaves English untouched, normalizes spacing."""

    def __init__(self, opencc_config: str = "s2twp", protect=None):
        try:
            from opencc import OpenCC  # opencc-python-reimplemented
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "opencc not installed. `pip install opencc-python-reimplemented`"
            ) from e
        self._cc = OpenCC(opencc_config)
        self._protect = list(protect or [])

    def _convert_cjk_only(self, text: str) -> str:
        # Convert each maximal CJK run as a whole phrase (preserves OpenCC's Taiwan-phrase layer);
        # English/ASCII spans between runs are left byte-identical.
        out = _CJK_RUN.sub(lambda m: self._cc.convert(m.group(0)), text)
        for simp, desired in self._protect:
            out = out.replace(simp, desired)
        return out

    def __call__(self, text: str) -> str:
        return normalize_cjk_spacing(self._convert_cjk_only(text))


def _iter_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


# Known-divergent segments: per-char conversion gives the wrong (glyph-only) result; whole-run s2twp
# must produce the Taiwan-phrase form on the right. Used by --selftest to prove the phrase layer works.
_SELFTEST = [
    ("软件更新", "軟體更新"),
    ("内存不足", "記憶體不足"),
    ("打印文件", "列印檔案"),  # note 文件->檔案: a Taiwan *vocabulary* swap per-char conversion can't do
    ("视频会议", "視訊會議"),
    ("鼠标和键盘", "滑鼠和鍵盤"),
]


def _selftest() -> int:
    proc = ZhTwPostProcessor("s2twp")
    bad = 0
    for src, want in _SELFTEST:
        got = proc(src)
        ok = got == want
        bad += not ok
        print(f"  {'ok ' if ok else 'FAIL'}  {src} -> {got}" + ("" if ok else f"   (want {want})"))
    # English must be byte-identical
    mixed = "请把 file 上传到 Google Drive"
    got = proc(mixed)
    if "file" not in got or "Google Drive" not in got:
        print(f"  FAIL  english not preserved: {got}"); bad += 1
    else:
        print(f"  ok    english preserved: {got}")
    print(f"[selftest] {'PASS' if bad == 0 else f'{bad} FAILURES'}", file=sys.stderr)
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", help="input file (TSV id<TAB>text, or plain text)")
    ap.add_argument("--out", dest="out", help="output file (same shape as input)")
    ap.add_argument("--opencc", default="s2twp", help="OpenCC config (default s2twp; s2tw = glyph-only)")
    ap.add_argument("--protect", help="TSV of simplified<TAB>desired post-OpenCC overrides (proper nouns)")
    ap.add_argument("--selftest", action="store_true", help="assert the s2twp phrase layer is working")
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())
    if not args.inp or not args.out:
        ap.error("--in and --out are required (or use --selftest)")

    protect = load_protect(args.protect) if args.protect else None
    proc = ZhTwPostProcessor(args.opencc, protect=protect)
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for line in _iter_lines(args.inp):
            if "\t" in line:
                uid, _, text = line.partition("\t")
                out.write(f"{uid}\t{proc(text)}\n")
            else:
                out.write(proc(line) + "\n")
            n += 1
    print(f"[zh_tw_postproc] {n} lines -> {args.out} (opencc={args.opencc})", file=sys.stderr)


if __name__ == "__main__":
    main()
