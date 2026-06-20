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

## Tier 0 — RESOLVED: the trainable artifact exists

> **Updated by research ([`docs/RESEARCH.md`](docs/RESEARCH.md)).** The earlier "ONNX-only /
> inference-only" assumption was **wrong**. Tiers 3–4 are **unblocked.**

The trainable artifacts are public:
1. **`streaming_exp/pretrained.pt` (2.56 GB)** on `GilgameshWind/X-ASR-zh-en` (HF) — the icefall
   checkpoint.
2. **The full recipe** at [`github.com/Gilgamesh-J/X-ASR`](https://github.com/Gilgamesh-J/X-ASR):
   `train.py`, `finetune.py`, `export-onnx-streaming.py`, `model.py`, `zipformer.py`, and the tokenizer
   `data/lang_5000_with_punctuation/bpe_punc.model`.

**Before any GPU time, smoke-test:** download the `.pt`, load it against the recipe `model.py`, and
confirm (a) it is the **punctuation** variant (vocab 5000-with-punct), (b) encoder dims/layers
(6 stacks/19 layers, 192·256·512·768·512·256) match the deployed ONNX, (c) it is a single averaged
checkpoint (export directly from it; no `--avg` epoch history). Confirm any "collected"-data license
terms before redistributing a derivative.

> Tokenizer decision: the vocab is **char+BPE, Simplified** (4000 CJK single chars + ~977 English BPE).
> **Keep the vocab and convert in post (Tier 1)** — extending it with Traditional tokens would force an
> output-embedding/joiner resize and a longer retrain for no acoustic gain.

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

### Data (see `docs/DATASETS.md` — corrected by research)
- **Taiwan-accent CS audio is scarce**: the only freely-downloadable Taiwan-accent zh-en code-switch
  audio is **NTUML2021** (~11h); the only large free Taiwan-Mandarin corpus is **Common Voice zh-TW**
  (~80h, CC0). **There is no public corpus of authentic Taiwan-accent zh-en CS audio** → manufacture it
  by **distilling Breeze-ASR-25 pseudo-labels** on real TW audio (see `docs/RESEARCH.md`).
  - **NOTE: TAT is Taiwanese *Hokkien*, not Mandarin** — do not use it here.
- **Code-switch (mainland/HK accent — switch-point coverage + retention, not TW accent)**: TAL-CSASR,
  ASCEND, CS-Dialogue, ASRU2019.
- **Retention set (non-negotiable)** — mirror the baseline's upstream `multi_zh_en` recipe:
  **LibriSpeech 960h (en) + TAL-CSASR (CS) + AISHELL-2 (mainland zh)** (or free proxies: LibriSpeech +
  ASCEND + AISHELL-1). icefall's `--use-mux` does this mixing natively if you have the cuts.

### Method — PEFT first (icefall ships all three paths)
- icefall provides three in-tree recipes for **this exact Zipformer2**, all with `--causal`/`--chunk-size`
  /`--left-context-frames`: `zipformer/finetune.py` (full/partial, `--init-modules encoder --use-mux
  --base-lr 0.0045`), `zipformer_adapter/` (bottleneck adapter, `--adapter-dim 8` ≈ 1.1% params,
  toggleable), and `zipformer_lora/` (`--use-lora 1 --lora-r 8`, encoder-only, **folds into weights at
  export → zero runtime cost**).
- **Start with LoRA (rank 8) or the bottleneck adapter on the encoder**, base frozen. For a transducer
  encoder, **PEFT is the proven anti-forgetting winner** (Whisper zh-en CS: full FT drove EN WER
  3.40→13.15; LoRA held 3.51). **The `jetson-tts` "task-vector over LoRA" preference does NOT carry over**
  — that LoRA failure was specific to the TTS CFM flow-matching decoder, not a transducer encoder.
- **Low LR** (LoRA `--base-lr 0.045`; full/partial `0.0045` ≈ 1/10), short schedule, frequent ckpts.
- Keep the model **causal / streaming** (`--causal 1`, matching chunk/left-context) — never fine-tune the
  offline variant. Recover X-ASR's exact encoder dims/chunk string so `--init-modules` prefix-matches.
- RNN-T (pruned transducer) loss as in the recipe; mix retention + TW data per batch (`--use-mux`).
- **Optional**: EWC / Synaptic Intelligence (memory-free, proven on RNN-T, ~4–5% rel WERR) if retention
  data is hard to obtain.

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
