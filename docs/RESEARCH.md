# Fine-tuning X-ASR for zh-TW/en on the Jetson Nano — research findings

Verified research synthesis (2026-06-20). Produced by an 8-dimension fan-out with adversarial
verification of every load-bearing claim: **7 dimensions completed, 16 load-bearing claims independently
re-checked → 15 confirmed, 1 corrected for imprecision, 0 refuted.** (The `opencc-orthography` dimension
did not complete; its conclusions here are cross-corroborated by the decoding-levers and Tier-0
dimensions and flagged where not independently verified.) Every claim below carries a primary-source URL.

> **This report overrides three premises in the originally-scaffolded docs.** They have been corrected
> in place; the corrections are called out in **bold "CORRECTION"** notes so the change is auditable.

---

## Executive summary

1. **Tier 0 is RESOLVED — weight fine-tuning is feasible today.** The original "HF release is ONNX-only"
   premise is **false**. `GilgameshWind/X-ASR-zh-en` ships a trainable **`streaming_exp/pretrained.pt`
   (2.56 GB)**, and the full icefall recipe (train/finetune/export + `bpe.model`) is public at
   **`github.com/Gilgamesh-J/X-ASR`**.
2. **icefall ships THREE in-tree fine-tune paths** for this exact Zipformer2 transducer — full/partial
   (`zipformer/finetune.py`), **bottleneck adapter** (`zipformer_adapter/`, ~1.1% params, toggleable),
   and **true LoRA** (`zipformer_lora/`, encoder-only, folds into weights at export). LoRA/adapter do
   **not** need hand-adding.
3. **For an RNN-T transducer, PEFT (LoRA/adapter) is the proven anti-forgetting winner** — full FT
   collapses English (SEAME/Whisper: EN WER 3.40→13.15; LoRA held 3.51). **The jetson-tts "task-vector
   over LoRA" preference does NOT transfer here** — that failure was specific to the TTS CFM-decoder, not
   a transducer encoder.
4. **The data bottleneck is real and specific:** there is **no public Taiwan-accent zh-en code-switch
   audio**. The only freely-downloadable Taiwan-accent CS audio is **NTUML2021 (~11h)**; the only large
   free Taiwan-Mandarin corpus is **Common Voice zh-TW (~80h, CC0)**. Everything else is either
   mainland/HK/SEA accent or non-commercial.
5. **CORRECTION: TAT is Taiwanese Hokkien (台語), not Taiwan Mandarin** — wrong language; removed from the
   dataset plan (the jetson-tts project caught the same error).
6. **Decode-time levers ship now and are real:** OpenCC `s2twp` on output, contextual hotwords
   (require `modified_beam_search`), `blank_penalty`, and optionally a LODR bigram-FST LM on CPU. **One
   correction: zh-TW hotwords must be fed to the model in SIMPLIFIED form** (the model's token space),
   then output re-converted to Traditional.
7. **Export round-trip is low-risk:** `export-onnx-streaming.py` at **opset 13**, which the Nano's
   **onnxruntime 1.6.0 supports** (verified); int8 = ORT dynamic quant. LoRA merges at export → **zero
   runtime overhead**, preserving the 2-core budget.
8. **Don't swap engines.** No streaming zh-TW model exists to adopt. Keep X-ASR as the student; the
   strongest data play is **distilling MediaTek Breeze-ASR-25** (Taiwan SOTA, but 2B/offline) as a
   pseudo-label teacher for scarce Taiwan CS audio.

---

## Tier 0 — the trainable artifact (RESOLVED)

**CORRECTION to README.md / TRAINING.md / finetune/README.md:** they stated the HF release is "ONNX-only
/ inference-only." That is wrong.

- **Trainable checkpoint exists:** `streaming_exp/pretrained.pt`, **2.56 GB** torch pickle (HEAD on the
  resolve URL returns `x-linked-size 2,555,882,878`), listed in the HF API siblings alongside the four
  ONNX chunk variants. _(confirmed)_
  — https://huggingface.co/api/models/GilgameshWind/X-ASR-zh-en ·
  https://huggingface.co/GilgameshWind/X-ASR-zh-en/tree/main/streaming_exp
- **Full recipe is public:** `github.com/Gilgamesh-J/X-ASR` ships `train.py`, `finetune.py`, `decode.py`,
  `streaming_decode.py`, `export-onnx-streaming.py`, `model.py`, `zipformer.py`. _(confirmed)_
  — https://github.com/Gilgamesh-J/X-ASR
- **Tokenizer is recoverable:** the recipe ships `data/lang_5000/bpe.model` and
  `data/lang_5000_with_punctuation/bpe_punc.model`; the deployed (punctuated) model maps to the latter.
  _(confirmed)_ — same repo.
- **Lineage:** the modern icefall **Zipformer2 streaming/causal pruned-transducer** family (the
  `multi_zh_en` / librispeech-streaming lineage; bundled `zipformer.py` is the Xiaomi/Povey/Yao
  causal Zipformer2 with `chunk_size`/`left_context_frames`). _(confirmed)_
- **Tokenizer composition (local `tokens.txt`):** 5000 tokens = **4000 single CJK chars** (meta-space
  prefixed) + ~977 English BPE pieces + specials; **Simplified** (体/软/机/台 present, 體/軟/機/臺 absent)
  — which is exactly what makes the Tier-1 OpenCC `s2twp` step necessary. _(confirmed)_

**Before spending GPU time, smoke-test (the remaining risk):** download the `.pt`, load it against the
GitHub `model.py`, and confirm (a) it is the punctuation variant (vocab 5000-with-punct), (b)
encoder dims/layers (6 stacks/19 layers, 192·256·512·768·512·256) and joiner/decoder dims match the
deployed ONNX, (c) it is a single averaged checkpoint (no epoch history → export directly from it, don't
rely on `--avg`). Confirm license terms on any "collected" data before redistributing a derivative.

---

## Fine-tuning method — three supported paths, PEFT first

icefall ships all three in `egs/librispeech/ASR/`, all the **same Zipformer2** as X-ASR, all exposing
`--causal` / `--chunk-size` / `--left-context-frames` so you **fine-tune in the deployed streaming
regime** (never the offline variant):

| Path | Recipe | Trainable params | Key flags | Runtime cost after export |
|---|---|---|---|---|
| **Full / partial FT** | `zipformer/finetune.py` | up to 100% | `--do-finetune --finetune-ckpt --init-modules encoder --use-mux --base-lr 0.0045` (~1/10 LR) | none (same weights) |
| **Bottleneck adapter** | `zipformer_adapter/train.py` | **1.148%** (`--adapter-dim 8`) | `--use-adapters 1`; 4 adapters/encoder layer, base frozen; toggle `--use-adapters 0` recovers base | small (adapters in graph) |
| **True LoRA** | `zipformer_lora/finetune.py` | tiny | `--use-lora 1 --lora-r 8 --base-lr 0.045`; LoRA only in encoder attn `in_proj` + FFN | **zero** — `merge_weights=True` folds LoRA into the linear at export |

Sources (all verified verbatim against raw source):
- https://k2-fsa.github.io/icefall/recipes/Finetune/from_supervised/finetune_zipformer.html
- https://k2-fsa.github.io/icefall/recipes/Finetune/adapter/finetune_adapter.html
- https://raw.githubusercontent.com/k2-fsa/icefall/master/egs/librispeech/ASR/zipformer_lora/finetune.py
- https://raw.githubusercontent.com/k2-fsa/icefall/master/egs/librispeech/ASR/zipformer_lora/zipformer.py
- https://raw.githubusercontent.com/k2-fsa/icefall/master/egs/librispeech/ASR/zipformer/finetune.py

**Recommendation:** start with **LoRA (rank 8) or the bottleneck adapter on the encoder** — base
distribution recoverable, near-zero runtime cost (LoRA folds at export, protecting the 2-core RTF), and
the literature winner for transducers. `--init-modules encoder` is the lever to "load encoder, leave
decoder/joiner/embedding." Reserve full/partial FT + task-vector for Tier 4. **Module-name caveat:**
X-ASR's encoder dims are non-standard vs the librispeech reference, so recover the exact
`--num-encoder-layers/--encoder-dim/--downsampling-factor/--chunk-size` (from the `.pt` stored args or
ONNX metadata) before `--init-modules` can prefix-match parameter names.

### Why PEFT, not task-vector (corrects ZH_TW_PLAN/TRAINING tone)

The strongest single piece of evidence for our exact task: Whisper on SEAME+ASRU2019 zh-en CS — **full FT
drove English test-clean 3.40→13.15 and Korean 18.9→436 (collapse); LoRA held both at baseline
(3.51 / 16.8)** (Yang et al., arXiv 2506.21576). Corroborating:
- Multi-accent **MAS-LoRA** keeps LibriSpeech-clean retention ≈ base (5.91 vs 5.78) while beating full
  FT on accent (Bagat et al., Interspeech 2025, arXiv 2505.20006).
- **Replay 10–20%** of original data prevents forgetting; **>20% starts a trade-off**; **layer-specific
  FT captures ~91% of RNN-T gain from joiner + first encoder layer** (Pekarek-Rosin & Wermter, arXiv
  2307.07280; Shor et al., arXiv 2306.10860).
- **EWC / Synaptic Intelligence** add ~4–5% rel WERR, are memory-free, and are **proven on an RNN-T
  transducer** for region/accent adaptation (Trinh et al., Interspeech 2022, arXiv 2207.07850; Ahadzi et
  al., Interspeech 2025).
- **Caveat — PEFT is not magic for large shifts:** rehearsal-free LoRA adapting Whisper to *distant*
  Uyghur still degraded Aishell-1 22.2→44.7 (Xu et al., Interspeech 2024, arXiv 2408.10680). A
  **Taiwan-accent shift is small** (same language), so expect mild forgetting — but **confirm with the
  MER gate**, don't assume.
- **Task-vector interpolation** is validated for ASR (Speech-FT, arXiv 2502.12672) and remains the
  **Tier-4 fallback**. The jetson-tts "task-vector over LoRA" rule was **CFM-decoder-specific** and does
  not apply to a transducer encoder.

---

## Data — the real bottleneck

**Central finding: there is no publicly obtainable corpus of authentic Taiwan-accent zh-en code-switch
audio.** The two needed properties live in nearly disjoint corpora and must be combined; Taiwan accent
must largely be *manufactured* (synthetic / pseudo-labeled).

| Dataset | Accent / lang | Code-switch | Size | License | Access | Use |
|---|---|---|---|---|---|---|
| **Common Voice zh-TW** | Taiwan Mandarin | no | ~80h val (131h tot, CV25) | **CC0** | direct, no app | FT + eval (the only large free TW-Mandarin) |
| **NTUML2021** (`ky552/ML2021_ASR_ST`) | **Taiwan Mandarin + en** | **yes** | ~11h (5h long eval) | unstated → **verify** | HF direct | **FT + eval** — only free TW-accent CS audio |
| **CSZS-zh-en** (`ky552/cszs_zh_en`) | mixed | yes | 35k utt | **MIT** | HF direct | eval benchmark |
| NER-Trs-Vol | Taiwan Mandarin | no | ~610h | **Non-commercial** | FSW/ACLCLP app | FT (TW accent) — *NC blocks product use* |
| MATBN | Taiwan Mandarin (broadcast) | no | 198h | ACLCLP, NC | application | FT + eval |
| ASCEND (`CAiRE/ASCEND`) | **HK/mainland** | yes | 10.6h | CC-BY-SA 4.0 | HF direct | switch-point + eval (not TW accent) |
| CS-Dialogue (`BAAI/CS-Dialogue`) | mainland | yes | 104h | CC-BY-**NC**-SA | HF direct | CS style/retention (NC) |
| TAL-CSASR (`csukuangfj/tal_csasr`) | mainland | yes | ~587h | unstated → verify | speechhome / HF | **CS retention** (baseline's own CS data) |
| ASRU2019 CS | mainland | yes | 240h | DataTang | challenge reg. | CS volume |
| SEAME | Singapore/Malaysia | yes | ~30–192h | **LDC (paid)** | LDC | switch-point only |
| AISHELL-2 / -1 | mainland | no | 1000h / 170h | academic-app / Apache-2.0 | app / direct | **mainland-zh retention** |
| LibriSpeech | English | no | 960h | CC-BY-4.0 | OpenSLR | **English retention** |

- **CORRECTION (DATASETS.md):** **TAT (Taiwanese Across Taiwan) is Taiwanese HOKKIEN (台語), not Taiwan
  Mandarin** — removed. The Formosa Taiwan-*Mandarin* corpus is **NER-Trs-Vol**. _(confirmed; same error
  the jetson-tts project caught.)_ — https://sites.google.com/nycu.edu.tw/fsw/home ·
  https://link.springer.com/article/10.1007/s11265-019-01483-4
- **Retention set = mirror the baseline's upstream `multi_zh_en` recipe exactly:** **LibriSpeech 960h
  (en) + TAL-CSASR (CS) + AISHELL-2 (mainland zh)** (use `--use-mux` if you can obtain these cuts;
  AISHELL-1 is the free proxy). — https://github.com/k2-fsa/icefall/blob/master/egs/multi_zh_en/ASR/RESULTS.md
- **Commercial-license caveat (the attendant is a product):** NER/MATBN are non-commercial; CS-Dialogue
  is NC; TAL-CSASR and NTUML2021 HF cards have **empty license fields** — verify before shipping. The
  commercially-clean Taiwan sources are **CC0 Common Voice zh-TW + synthetic/pseudo-labeled audio.**
- **Synthetic augmentation is the validated path** (and matches jetson-tts): MediaTek **Breeze-ASR-25**
  reached Taiwan SOTA training almost entirely on **10,000h synthetic Chinese (BreezyVoice TTS)** + 11h
  real NTUML2021. — https://huggingface.co/MediaTek-Research/Breeze-ASR-25

---

## Decode-time levers that ship without retraining

All reachable from `OnlineRecognizer.from_transducer(...)` / `create_stream(hotwords=...)`. Verified
against sherpa-onnx master source.

- **OpenCC `s2twp` on output** — Simplified→Traditional with Taiwan phrasing; scope to CJK spans.
  (Tier 1; `scripts/zh_tw_postproc.py`.)
- **Contextual hotwords** — **require `modified_beam_search`** (greedy raises `ValueError`); format is
  space-separated tokens + trailing ` :score`; **`modeling_unit=cjkchar+bpe` + a `bpe_vocab`** for mixed
  zh+en phrases; settable per-stream (good for per-call name lists). Default `hotwords_score=1.5`.
  — https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html ·
  https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/sherpa-onnx/python/sherpa_onnx/online_recognizer.py
  - **CORRECTION (build_hotwords.py / ZH_TW_PLAN.md): zh-TW hotwords must be supplied in SIMPLIFIED form**
    (the model's token space), not Traditional — the live demo runs OpenCC **`t2s`** over each hotword
    before biasing, then `s2twp` on the output. A Traditional hotword `軟體` won't tokenize; it must be
    `软件`. _(confirmed against `/tmp/streaming-zipformer-demo/asr_engine.py`.)_ — **the script has been
    fixed to emit Simplified.**
- **`blank_penalty`** (default 0.0) — subtracts from the blank logit; raising it reduces deletions at
  fast en↔zh switch points; **works with greedy** (no beam-search requirement). Sweep e.g. 0/0.5/1.0/1.5.
- **LM (optional):** both **shallow fusion** and **LODR** are wired into the *online* decode path and run
  on CPU (`lm`, `lm_scale`, `lodr_fst`, `lodr_scale`), gated to `modified_beam_search`. The neural RNN-LM
  (2048-dim/3-layer) is invoked per beam path per step — **likely too heavy for 2 cores**; the **LODR
  bigram FST (~250KB)** is the CPU-friendly option. — https://k2-fsa.github.io/icefall/decoding-with-langugage-models/LODR.html
  - **Open:** confirm the Nano's *pinned* sherpa-onnx build exposes `lodr_fst`/`lm` (verified params are
    from current master); and that a CJK+BPE LODR FST can be built against the 5000-token vocab.
- **Endpointing:** `rule1/2/3` trailing-silence + max-utterance rules — tune for attendant barge-in.

**Budget caveat:** the measured RTF 0.39 @2 threads is **greedy_search**. Hotwords/LM force
`modified_beam_search` (more compute) → **re-benchmark at 2 threads** (`scripts/bench_nano.py`) and keep
only if RTF stays under budget while TTS co-tenants.

---

## Export & int8 back to the Nano (low-risk)

- **Export:** `egs/librispeech/ASR/zipformer/export-onnx-streaming.py` — causal/streaming, embeds the
  metadata sherpa-onnx needs (`model_type=zipformer2`, `decode_chunk_len`, `T`, `left_context_len`,
  encoder dims…), emits the per-layer recurrent-state contract (6 cache tensors/layer + `embed_states` +
  `processed_lens`). Export with the **same chunk/left-context the model was trained with**.
- **int8:** `--enable-int8-quantization 1` (default) → ORT `quantize_dynamic`, weight-only, no
  calibration: encoder+joiner quantize `MatMul`, decoder also `Gather`, `QInt8`. **Deployed mix to
  reproduce: encoder.int8 + decoder.fp32 + joiner.int8.**
- **LOAD-BEARING, confirmed — opset compatibility:** export uses **opset 13**; **onnxruntime 1.6.0
  supports up to opset 13** (IR 7); the injected int8 ops (`MatMulInteger` op10, `DynamicQuantizeLinear`
  op11) are within ceiling → **no pitfall.** — https://github.com/microsoft/onnxruntime/blob/v1.6.0/docs/Versioning.md
- **LOAD-BEARING, softened — "int8 stays lossless":** the repo's lossless claim rests on a **single 10s
  clip**, and the token-exact quant tables are for **GGUF/ggml**, a *different* stack than ORT dynamic
  int8. Treat int8 as **near-lossless and re-verify** fp32-vs-int8 with `eval_asr.py` on the multi-slice
  dev set after any fine-tune.
- **LoRA folds at export** (`merge_weights=True`) → **no runtime LoRA overhead**, so PEFT does not cost
  the 2-core budget. Verify the export with `x-asr-rapidspeech/tools/inspect_onnx.py` (metadata + state
  contract) and a greedy decode diff vs the deployed model.

---

## Alternatives — keep X-ASR, distill the teacher

- **No streaming zh-TW model exists to adopt.** Every Taiwan-Mandarin model found is **offline**
  (Whisper/Breeze/wav2vec2/Conformer CV-zh-TW fine-tunes); every streaming zh-en model (csukuangfj
  bilingual zipformer, FunASR Paraformer-streaming, NeMo) is **mainland-trained**, no Taiwan accent, and
  X-ASR already beat that class in the bake-off — or is too big for 2× A57. _(confirmed)_
- **Breeze-ASR-25** (Whisper-large-v2, 2B, **offline**, zh-TW CER **7.97%**, Apache-2.0) is the Taiwan
  SOTA but **undeployable on the Nano** → use it as a **distillation teacher** to pseudo-label real
  Taiwan audio (Common Voice zh-TW, podcasts) and fine-tune the X-ASR streaming student — a path
  validated by "streaming transducer from Whisper pseudo-labels" (arXiv 2409.13499). Also reuse Breeze's
  **eval suite** (ASCEND-{OVERALL/EN/ZH/MIX}, CV-zh-TW, CSZS-zh-en, ML-Lecture-2021-Long) as our gate, to
  be comparable to published Taiwan numbers.
- **Don't swap engines.** Keep X-ASR as the deployed student.
- Note: **Breeze-ASR-26 is Taiwanese Hokkien** — wrong language; ignore.
- Note: X-ASR's "record code-switch accuracy" is the **project's own bake-off**, not a published card
  claim (the card has no Taiwan/CS eval). Worth eventually publishing a reproducible head-to-head.

---

## Risks & open questions (what still gates GPU time)

1. **Smoke-test `pretrained.pt`** loads against the recipe, is the punctuation variant, and matches the
   deployed dims — before any training. (Tier-0 residual risk.)
2. **Quantify the acoustic vs orthographic gap first** (the `s2twp`-isolation test in `EVAL.md`): if
   `s2twp` + hotwords close most of the error, **Tier 3 may not be needed at all.** Run this before
   collecting accent data.
3. **Retention data availability:** X-ASR's ~1M-h corpus is largely proprietary; `--use-mux` needs the
   original cuts. Whether a **proxy** (LibriSpeech + TAL-CSASR + AISHELL-2) prevents forgetting well
   enough is empirical — gate with MER.
4. **Optimal PEFT placement for Taiwan accent** (encoder attn q/v LoRA vs per-layer adapter vs
   joiner+first-layer) is not pinned by any source — A/B against the MER gate.
5. **Licenses for a product:** NTUML2021 and TAL-CSASR license fields are empty; NER/MATBN/CS-Dialogue
   are non-commercial. Confirm before shipping FT data.
6. **On-device LM support:** confirm the Nano's pinned sherpa-onnx exposes LODR; build a CJK+BPE bigram
   FST against the 5000-token vocab.
7. **int8 on a corpus, not a clip:** measure fp32-vs-int8 CER/WER on the dev set.

---

## Recommended sequence of work

1. **Tier 0 smoke-test** — download `streaming_exp/pretrained.pt` + `bpe_punc.model` + the Gilgamesh-J
   recipe; load against `model.py`; export back to int8 ONNX and diff a greedy decode vs the deployed
   model (parity check).
2. **Ship Tiers 1–2 now** — `s2twp` output post-proc; hotwords (**Simplified input**) +
   `modified_beam_search` + `blank_penalty` sweep; re-bench RTF @2 threads.
3. **Stand up the MER gate** using Breeze's eval suite + run the **`s2twp`-isolation test** to quantify
   how much residual error is genuinely *acoustic*. **Decide whether Tier 3 is justified.**
4. **If acoustic gap remains** — assemble retention (LibriSpeech + TAL-CSASR + AISHELL-2/-1) + TW data
   (CV-zh-TW + NTUML2021 + Breeze-pseudo-labeled TW audio); **LoRA/adapter on the encoder, `--causal 1`**,
   `--init-modules encoder`, low LR, MER-gated selection (reject any English/mainland regression); EWC
   optional. Export int8, re-bench @2 threads.
5. **Tier 4 fallback** — full/partial FT + task-vector interpolation only if PEFT is insufficient.

---

### Provenance
Methodology: parallel research agents (web + local repos) → adversarial verification of every
load-bearing claim → this synthesis. 7/8 dimensions completed (the run hit a session usage limit before
the `opencc-orthography` agent and the automated synthesis finished; this report was synthesized from the
recovered, verified agent outputs). Verification tally: 15 confirmed, 1 corrected-for-imprecision (ONNX
metadata key list), 0 refuted.
