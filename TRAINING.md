# Training — making X-ASR better at zh-TW / en under the attendant's 2-core budget

The baseline ([`README.md`](README.md)) is the **record-accuracy** zh-en streaming zipformer2 transducer
(X-ASR), already real-time on the Nano at **int8 / sherpa-onnx CPU / 2 threads**. This document is the
recipe for making it **better for Taiwan** without losing what makes it the baseline: English quality,
mainland-zh quality, code-switch robustness, and the 2-core real-time budget.

The work is laddered. **Lower tiers ship today with no GPU and no retrain; only the top tier touches
weights.** Climb only as far as the error analysis justifies — most of the perceived "wrong for Taiwan"
gap is orthography + vocabulary, which the bottom two rungs close.

```
Tier 0  Confirm/recover a trainable artifact            (prerequisite for Tiers 3–4 only)
Tier 1  Orthography: OpenCC s2twp post-conversion        ◄ ships today, zero retrain
Tier 2  Vocabulary + code-switch: contextual hotwords    ◄ ships today, zero retrain
        + decoding (modified_beam_search, blank penalty)
Tier 3  Encoder adaptation: LoRA/adapter FT on TW audio  ◄ the real fine-tune (data-bound)
Tier 4  Full/partial FT + accent vector interpolation    ◄ last resort; high regression risk
```

---

## Tier 0 — Confirm the trainable artifact (do this before any GPU time)

The HF release ships the **sherpa-onnx ONNX export** (encoder/decoder/joiner + `tokens.txt`), which is
**inference-only**. Tiers 3–4 need one of:

1. **The original icefall checkpoint** — `pretrained.pt` / `epoch-*.pt`, the BPE/char tokenizer that
   produced the 5000-token vocab, and the model `config` (encoder dims, chunk/left-context, causal=True).
   Ask the X-ASR author; check the icefall recipe lineage referenced in
   [`ref/`](ref/) and `x-asr-rapidspeech/ref/icefall/`.
2. **A re-train from the icefall recipe** to reproduce a trainable equivalent, then continue from there.
3. **Nothing** — Tiers 3–4 are blocked. **Tiers 1–2 still deliver the bulk of the zh-TW win**, which is
   why they are built first and depend on no checkpoint.

> Sanity check the tokenizer first: the vocab mixes CJK chars with English BPE pieces. Whether to keep
> Simplified char tokens (and convert in post) or to extend the vocab with Traditional tokens is decided
> here — extending the vocab forces an output-embedding/joiner resize and a longer retrain, so the
> default is **keep the vocab, convert in post (Tier 1)**.

---

## Tier 1 — Orthography (ships today, zero retrain)

X-ASR emits **Simplified** Chinese. Taiwan reads **Traditional**, and crucially with **Taiwan word
choices**, which is `s2twp` (phrase-aware), not the character-only `s2tw`:

```bash
# in the streaming loop, convert each finalized segment before display/sending to the LLM
python scripts/zh_tw_postproc.py --in hyp_simplified.tsv --out hyp_zhtw.tsv --opencc s2twp
```

- `s2twp` does `软件→軟體`, `内存→記憶體`, `打印→列印`, `视频→影片`, `鼠标→滑鼠` — phrase-level Taiwan
  vocabulary, not just glyph mapping.
- Apply it **only to CJK spans**; never touch the English BPE output. `scripts/zh_tw_postproc.py` already
  scopes conversion to CJK and reuses the CJK-spacing/punctuation normalization from the upstream sherpa
  streaming frontend (`ref/`).
- Cost: a few hundred µs per segment on the A57; **no extra core, no model change.** This is the STT
  analog of `jetson-tts`'s shipped TW-readings lexicon swap.

**This rung alone removes most "looks mainland" complaints.** Measure its effect with `eval_asr.py`
scored against **Traditional** references (`docs/EVAL.md`).

---

## Tier 2 — Vocabulary & code-switch biasing (ships today, zero retrain)

For Taiwan-specific named entities, brand/product terms, and the attendant's own domain vocabulary,
bias the decoder rather than retrain it. sherpa-onnx supports **contextual hotwords** on this exact model.

```bash
# 1. build a boosted hotwords file from the curated CN→TW term list
python scripts/build_hotwords.py \
    --terms data/tw_terms/cn_tw_terms.tsv \
    --tokens ref/tokens.txt \
    --boost 2.0 \
    --out data/tw_terms/hotwords.txt

# 2. enable in sherpa-onnx (decoding must be modified_beam_search for hotwords to apply)
#    OnlineRecognizer(..., decoding_method="modified_beam_search",
#                     hotwords_file="data/tw_terms/hotwords.txt", hotwords_score=2.0)
```

Levers, in order of cost:

1. **Hotwords / contextual biasing** — boost attendant domain terms and TW entities. Multi-token zh-TW
   and mixed zh+en phrases are supported (the live demo already does this). Tune `hotwords_score`
   (start 1.5–2.5); too high causes over-triggering.
2. **`modified_beam_search` over `greedy_search`** — required for hotwords; also reduces code-switch
   boundary errors at the switch point. Costs a little latency — **re-measure at 2 threads** (`bench_nano.py`);
   keep it only if RTF stays < ~0.6 so TTS still fits.
3. **Blank penalty** — small negative blank bias reduces deletions in fast code-switch; sweep on the dev
   set, watch for insertion regressions.

Tiers 1+2 are **deployable on the Nano as-is** and need no checkpoint. Lock in their gains and
**re-run the error analysis** before deciding whether Tier 3 is even warranted.

---

## Tier 3 — Encoder adaptation on Taiwan audio (the real fine-tune)

Only the **acoustic accent** gap (Taiwan-Mandarin phonetics, Taiwanese-influenced Mandarin) needs
weights to move. The danger is the same one `jetson-tts` documented for accent FT: pulling the model
toward TW audio **degrades English and mainland-zh / code-switch**. So adapt narrowly and gate hard.

### Data (see `docs/DATASETS.md`)
- **Taiwan Mandarin**: TAT (Taiwanese Across Taiwan), Formosa Speech (FSR), Common Voice **zh-TW**.
- **Code-switch**: ASCEND, SEAME (zh-en switching — accent differs, but boundary/style coverage helps).
- **Retention set (non-negotiable)**: hold in a fixed fraction of the *original* English + mainland-zh +
  code-switch distribution every batch. This is the analog of `jetson-tts`'s English-retention set;
  without it the model forgets en.

### Method — adapter / LoRA, low LR, frozen where possible
- Train **LoRA or a bottleneck adapter on the zipformer encoder** (and optionally the joiner), keeping
  the bulk of the encoder, the decoder (predictor), and the embedding **frozen**. Full fine-tune is
  Tier 4 and the last resort — in `jetson-tts`, LoRA on the wrong module *and* full FT both collapsed
  content; the lesson is **adapt the fewest parameters that move the accent, and verify constantly**.
- **Low LR** (≈1e-5 region), short schedule, frequent checkpoints.
- Keep the model **causal / streaming** (chunk + left-context config unchanged) — never fine-tune the
  non-streaming variant and hope it exports back; train in the deployed streaming regime.
- RNN-T (pruned transducer) loss as in the icefall zipformer recipe; mix retention + TW data per batch.

### Gate — MER-selected, not loss-selected
Pick the checkpoint by **held-out MER on a multi-distribution dev set**, never by training loss:

| dev slice | metric | must-not-regress |
|---|---|---|
| Taiwan Mandarin | CER | improves (the point) |
| English | WER | ≤ baseline + ε |
| mainland zh | CER | ≤ baseline + ε |
| zh-en code-switch | MER + boundary-F1 | ≤ baseline + ε |

A checkpoint that improves TW CER but regresses English WER is **rejected** — exactly the rule that made
`jetson-tts` choose a conservative blend over a stronger-but-broken one.

### Export back to the deployed runtime
1. Export the adapted streaming model to ONNX via the icefall zipformer **streaming** export
   (`export-onnx-streaming.py`), preserving the chunk/left-context contract and metadata.
2. **int8-quantize** the ONNX (sherpa-onnx dynamic quantization) — int8 is the deployed format and the
   one the 2-core budget was measured against.
3. **Re-benchmark on the real Nano at 2 threads** (`bench_nano.py`). A quality win that breaks the
   real-time budget is not a win for the attendant — it must stay under real-time while TTS runs too.

---

## Tier 4 — Full / partial FT + accent-vector interpolation (last resort)

If adapters can't move the accent enough, mirror the `jetson-tts` endgame: full (or layer-scoped)
fine-tune to get θ_ft, then **interpolate the task vector** θ = θ_base + α·(θ_ft − θ_base) and sweep α on
the MER gate above, choosing the largest α that keeps English/mainland-zh within ε. Expect this to be
**data-bound, not tuning-bound**: the ceiling is set by how much *labeled Taiwan-accent code-switch
audio in a compatible style* you have, not by the optimizer. Budget accordingly, and keep Tiers 1–2
shipped in the meantime.

---

## What success looks like

- **Tier 1–2**: Taiwan-correct *orthography and vocabulary* in the live transcript, zero retrain, no
  change to the 2-thread RTF — deployable now.
- **Tier 3**: lower **CER on Taiwan-accented speech** with **English WER and code-switch boundary-F1
  held at baseline**, still int8, still real-time at 2 threads on the Nano.
- **Always**: the model leaves the 3rd and 4th A57 cores for TTS and the rest of the attendant.

See [`docs/EVAL.md`](docs/EVAL.md) for the exact metrics and `scripts/eval_asr.py` / `scripts/bench_nano.py`
for the harness.
