#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tier-1 zh-TW post-processing: Simplified -> Traditional (Taiwan phrasing) + CJK normalization.

X-ASR emits Simplified Chinese with mainland word choices. Taiwan reads Traditional with Taiwan
phrasing, which is OpenCC `s2twp` (phrase-aware), not the glyph-only `s2tw`. Conversion is scoped to
CJK spans so English BPE output is never touched. Costs no model change and no extra CPU core — this is
the dominant "looks wrong for Taiwan" fix (see docs/ZH_TW_PLAN.md, TRAINING.md Tier 1).

Usage:
    # one-shot over a TSV (id<TAB>text) or plain-text file
    python scripts/zh_tw_postproc.py --in hyp.tsv --out hyp_zhtw.tsv --opencc s2twp
    # or as a library in the streaming loop:
    #   from zh_tw_postproc import ZhTwPostProcessor; p = ZhTwPostProcessor(); p(text)
"""
import argparse
import re
import sys

# CJK ranges / punctuation, mirroring the upstream sherpa streaming frontend
# (x-asr-rapidspeech/ref/sherpa_streaming_infer.py) so spacing matches the deployed runtime.
_CJK_RANGE = r"㐀-䶿一-鿿豈-﫿"
_CJK_PUNCT = re.escape("，。！？；：、（）《》〈〉【】「」『』“”‘’")
_ASCII_PUNCT_NO_LEADING_SPACE = re.escape(",.!?;:%)]}")
_CJK_CHAR = re.compile(rf"[{_CJK_RANGE}]")


def normalize_cjk_spacing(text: str) -> str:
    """Drop spurious spaces around CJK characters and punctuation."""
    text = re.sub(rf"(?<=[{_CJK_RANGE}])\s+(?=[{_CJK_RANGE}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_RANGE}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_RANGE}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_PUNCT}])\s+(?=[{_CJK_PUNCT}])", "", text)
    text = re.sub(rf"\s+(?=[{_ASCII_PUNCT_NO_LEADING_SPACE}])", "", text)
    return text


class ZhTwPostProcessor:
    """Callable: converts CJK runs to Taiwan Traditional, leaves English untouched, normalizes spacing."""

    def __init__(self, opencc_config: str = "s2twp"):
        try:
            from opencc import OpenCC  # opencc-python-reimplemented
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                "opencc not installed. `pip install opencc-python-reimplemented`"
            ) from e
        self._cc = OpenCC(opencc_config)

    def _convert_cjk_only(self, text: str) -> str:
        # OpenCC on a mixed string is safe for English, but converting only CJK spans guarantees
        # English bytes are byte-identical and keeps conversion cost proportional to CJK content.
        return _CJK_CHAR.sub(lambda m: self._cc.convert(m.group(0)), text)

    def __call__(self, text: str) -> str:
        return normalize_cjk_spacing(self._convert_cjk_only(text))


def _iter_lines(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="input file (TSV id<TAB>text, or plain text)")
    ap.add_argument("--out", dest="out", required=True, help="output file (same shape as input)")
    ap.add_argument("--opencc", default="s2twp", help="OpenCC config (default s2twp; s2tw = glyph-only)")
    args = ap.parse_args()

    proc = ZhTwPostProcessor(args.opencc)
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
