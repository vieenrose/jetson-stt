# zh-TW adaptation plan — what's a post-process and what's a fine-tune

This is the STT mirror of `jetson-tts`'s zh-TW analysis. There, **readings (字音) shipped without
retrain** but **accent (腔調) was data-bound and out of scope**. The STT split is the same shape, just
relocated: the *written form* is fixed for free; the *acoustic accent* is the data problem.

## The four layers, from cheapest to dearest

### 1. Orthography (字形) — ✅ ships today, zero retrain
X-ASR outputs **Simplified**. Taiwan writes **Traditional with Taiwan word choices** → OpenCC **`s2twp`**
(phrase-aware; `s2tw` is glyph-only and misses 软件→軟體). Scope conversion to CJK spans only; never
touch English. → `scripts/zh_tw_postproc.py`. This is the dominant "looks wrong for Taiwan" fix and it
costs no model change and no extra core.

### 2. Vocabulary / entities (詞彙) — ✅ ships today, zero retrain
Taiwan term choices (軟體, 滑鼠, 計程車, 馬鈴薯, 行動電話…) and attendant-domain / named entities →
**contextual hotword biasing** in sherpa-onnx with `modified_beam_search`. Curated list:
`data/tw_terms/cn_tw_terms.tsv` → `scripts/build_hotwords.py`. `s2twp` already handles most generic
vocabulary; hotwords are for **entities and domain terms** OpenCC can't know.
> **Verified caveats (`RESEARCH.md`):**
> 1. **Bias in SIMPLIFIED form** (the model's token space) — a Traditional hotword `軟體` won't tokenize
>    and silently never biases; it must be `软件`. `build_hotwords.py` emits Simplified (via the `cn_term`
>    column or OpenCC `t2s`).
> 2. **Pure-CJK hotwords (zh-TW names/terms) work immediately** via the `cjkchar` path. **English/mixed
>    hotwords additionally need `modeling_unit=cjkchar+bpe` + a `bpe.vocab` — which X-ASR does NOT ship**
>    (only `tokens.txt`, not a valid `bpe.vocab`). To bias English phrases, rebuild `bpe.vocab` from the
>    recipe's `bpe.model` first. Start with pure-CJK hotwords.
> 3. Hotwords force `modified_beam_search` (greedy raises `ValueError`) → **re-bench RTF @2 threads**.

### 3. Code-switch boundaries (轉換點) — ✅ ships today, zero retrain
Errors at the en↔zh switch point are a *decoding* problem before they're a *model* problem:
`modified_beam_search`, hotwords spanning the boundary, and a small blank-penalty sweep. Re-measure RTF
at 2 threads after each — the attendant's budget is the hard constraint.

### 4. Acoustic accent (腔調) — ⚠️ data-bound fine-tune
Recognizing strongly **Taiwan-accented / Taiwanese-influenced Mandarin** (reduced retroflex, no erhua,
TW prosody) is the only layer that needs weights to move. This is **Tier 3 of `TRAINING.md`** and carries
the regression risk `jetson-tts` proved out empirically: naïve fine-tuning toward TW audio **degrades
English and mainland-zh / code-switch**. Mitigation is structural, not a knob:
- adapt the **fewest parameters** (LoRA/adapter on the encoder), freeze the rest;
- keep an **English + mainland-zh + code-switch retention set** in every batch;
- **select the checkpoint by multi-distribution MER**, rejecting any English/mainland regression;
- and remember it is **gated by data availability** (labeled TW-accent code-switch audio), not by tuning.

## Order of operations
Ship 1–3 first; they need no checkpoint and close most of the gap. **Re-run error analysis** before
spending GPU time on 4 — quantify how much of the remaining error is genuinely *acoustic* (model hears
it wrong) vs *orthographic/lexical* (model heard it right, wrote it mainland). Only the former justifies
Tier 3. See `docs/EVAL.md` for how to split those buckets (score CER against Traditional refs both with
and without `s2twp` to isolate orthography from acoustics).

## Cross-reference
- Fine-tune mechanics & gating: [`../TRAINING.md`](../TRAINING.md)
- Corpora: [`DATASETS.md`](DATASETS.md)
- Metrics: [`EVAL.md`](EVAL.md)
- TTS-side precedent (accent is data-bound): `jetson-tts/docs/TW_ACCENT_RESEARCH.md`
