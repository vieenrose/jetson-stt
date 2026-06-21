#!/usr/bin/env python3
"""Build the native-Traditional model: deployed X-ASR + s2twp-relabeled tokenizer (no retraining).
Usage: build_native_strong.py <deployed_model_dir> <out_dir>
Matches `deployed + s2twp` (0.068 Trad CER) but emits Traditional directly with no runtime OpenCC."""
import sys, os, shutil, re
from opencc import OpenCC
s2twp = OpenCC("s2twp"); CJK = re.compile(r"[㐀-鿿]")
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
for f in ("encoder.int8.onnx", "decoder.onnx", "joiner.int8.onnx"):
    shutil.copy(f"{src}/{f}", f"{dst}/{f}")
shutil.copy(f"{src}/tokens.txt", f"{dst}/tokens_simplified.txt")  # keep original for side-by-side
out = []
for l in open(f"{src}/tokens.txt", encoding="utf-8"):
    l = l.rstrip("\n"); p = l.rsplit(" ", 1)
    if len(p) != 2: out.append(l); continue
    tok, i = p; pre = "▁" if tok.startswith("▁") else ""; body = tok[len(pre):]
    if body and all(CJK.match(c) for c in body): body = s2twp.convert(body)  # bake s2twp into the token surface
    out.append(f"{pre}{body} {i}")
open(f"{dst}/tokens.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print(f"wrote {dst}: native Traditional tokenizer ({len(out)} tokens) + deployed ONNX")
