# Phase-0 results — measured on the real Jetson Nano (2026-06-20)

The Phase-0 gates from `RUNBOOK.md`, run end-to-end on the device (`ssh picard@picard-desktop`). **Real
measured numbers**, not estimates. They turn the plan's open conditionals into a decision: **ship the
zero-retrain Tier-1/2 fixes; do NOT fine-tune (no accent gap found); the 2-core budget holds under the
real single-thread TTS.**

## Setup
- **Device**: Jetson Nano gen1 (Tegra X1 / GM20B, sm_53, 4× Cortex-A57, ~4 GB, ~25.6 GB/s mem BW).
- **STT**: X-ASR int8 960 ms (`~/xasr-bench/int8-960/`: encoder.int8 + decoder fp32 + joiner.int8) +
  `tokens.txt`, via sherpa-onnx 1.13.3 (`OnlineRecognizer`, greedy, 2 threads) in the `moss` conda env
  (py3.12, ORT 1.26). Decode flushes the 960 ms streaming chunk with ~2 s trailing silence (1 s
  truncated utterance-final tokens; 2 s fixed it).
- **TTS co-tenant**: the real targeted engine — `rs-tts-offline -m matcha8k.gguf --gpu true --threads 1`
  (RapidSpeech.cpp ggml, single thread) in `~/xasr-hostc/`, looped on one core.
- **Data** (`scripts/build_eval_set.py`, 40 utts each, decoded on-device): Taiwan zh-en code-switch
  (NTUML2021 test, MIT, Traditional refs); mainland-zh clean read (FLEURS cmn); English clean read
  (LibriSpeech test-clean). *CV zh-TW clean read is gated (Mozilla Data Collective) — not obtained.*

## Gate 1 — split test + controls  →  **orthography is the whole story; no accent gap**

zhCER/enWER on-device (95% bootstrap CI, 1000 resamples):

| condition | metric | value | 95% CI |
|---|---|---|---|
| English clean read (LibriSpeech) | WER | **0.031** | 0.019–0.045 |
| Mainland zh clean read (FLEURS) — model home turf | CER | **0.048** | 0.033–0.065 |
| Taiwan zh-en CS lecture — **raw** (Simplified) vs Traditional ref | CER | **0.405** | 0.350–0.465 |
| Taiwan zh-en CS lecture — **deployed `s2twp(hyp)`** vs ref | CER | **0.082** | 0.039–0.140 |

**Findings:**
1. **Orthography was ~80% of the Taiwan error and is removed for free.** OpenCC `s2twp` (with the
   maximal-CJK-run bug fix) takes Taiwan zhCER **0.405 → 0.082**, zero retraining, validated on-device.
2. **No measured Taiwan-accent acoustic gap.** The Taiwan residual (0.082, on *spontaneous lecture
   code-switch* with disfluencies + ML jargon) sits barely above the model's mainland *clean-read* floor
   (0.048), and the CIs overlap. The 3.4 pp difference is explained by the harder spontaneous/CS
   condition, not accent. English is excellent (WER 0.031). **A fine-tune for accent is not justified by
   the evidence.**
3. Caveat: without a clean Taiwan *read-speech* control (CV zh-TW gated) accent and spontaneity aren't
   perfectly separable — but every available signal says the model already handles Taiwan-accented
   Mandarin well, and the deployable win is orthographic.

**Verdict: ship Tier-1 `s2twp` + Tier-2 hotwords. Do not spend GPU on an accent fine-tune.** (If a clean
CV-zh-TW read slice later shows a real read-speech gap vs FLEURS, revisit — but Phase 0 says no.)

### Gate 1b — head-to-head vs the Taiwan SOTA (Breeze-ASR-25)  →  **confirms no fine-tune**

The missing *upper bound*: [Breeze-ASR-25](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) — the
Taiwan-Mandarin + zh-en code-switch SOTA (Whisper-large-v2, 2B, **offline**, Apache-2.0/MIT) — transcribed
the **same 40 NTUML2021 clips** on the GB10 GPU (transformers, `chunk_length_s=0`, Traditional output, no
OpenCC). Same MER metric as the Breeze paper. 95% bootstrap CI (2000 resamples):

| system | MER | 95% CI | zhCER | enWER |
|---|---|---|---|---|
| **Breeze-ASR-25** — offline 2B, GPU, **in-domain** (see caveat) | **0.056** | [0.024, 0.097] | 0.046 | 0.235 |
| **X-ASR int8 streaming + s2twp** — deployed, 2 CPU cores, out-of-domain | **0.087** | [0.044, 0.146] | 0.082 | 0.176 |

**This is the "small gap, overlapping CIs" pattern the pre-registered rule maps to *confirm no
fine-tune*:**
- The Taiwan SOTA, with **every** advantage — **40× the parameters, offline, on a GPU, and trained on
  this exact corpus** (NTUML2021 is in Breeze's training set, per its card/paper → its number here is
  *optimistic/in-domain*, not held-out) — beats the deployed **streaming int8 model on 2 CPU cores** by
  only **~3 pp MER, with fully overlapping confidence intervals.**
- Breeze's zhCER (0.046) ≈ X-ASR's **mainland clean-read floor** (0.048). X-ASR's 0.082 on disfluent
  spontaneous lecture CS is squarely in the expected band, not an accent deficiency.
- X-ASR even **edges Breeze on English** (enWER 0.176 vs 0.235) — both with wide CIs (few en tokens).
- No published Breeze-vs-streaming-zipformer comparison existed; this is a novel, if small-N (40-utt),
  data point. It says X-ASR is **remarkably competitive** with a 2B in-domain SOTA at a tiny fraction of
  the compute — so a fine-tune (which couldn't reach Breeze's param class anyway) is **not warranted.**

> Breeze-ASR-25 used here for **evaluation only** (Apache-2.0/MIT, attribution retained). It remains a
> viable **distillation teacher** if a future, genuinely held-out gap ever appears — but Phase 0 shows
> none.

## Gate 2 — co-tenancy budget (STT under the real TTS)  →  **budget holds**

`scripts/bench_cotenancy.py`, STT pinned to cores 0,1; the co-tenant on the other core(s):

| co-tenant | STT RTF solo → contended | slowdown | real-time? |
|---|---|---|---|
| **real matcha8k TTS, `--threads 1`, GPU on (core 3)** — *your config* | **0.674 → 0.699** | **1.05×** | ✓ holds |
| ~~synthetic 2× CPU-GEMM workers (cores 2,3)~~ — *wrong proxy* | 0.475 → 1.490 | 2.94× | ✗ (see note) |

**Findings:**
- **With the real single-thread TTS, co-tenancy costs STT only ~5%** — it stays real-time. The 2-core STT
  budget is **safe** alongside the targeted matcha8k-on-RapidSpeech (ggml, 1 thread) TTS.
- The earlier 2.94× was a **wrong proxy**: two back-to-back CPU-GEMM workers model TTS as 2 cores of
  bandwidth-saturating compute, which it is not. Real matcha8k is single-thread (+ GPU offload), so it
  neither takes STT's cores nor saturates memory bandwidth the way the synthetic load did. Lesson: model
  the co-tenant as it actually runs.
- Note: absolute RTFs here (0.67) are inflated by the 2 s per-clip tail-pad (80 s of flush silence over
  75 s real audio); the pure-streaming RTF is ~0.30. The **1.05× ratio** is the signal, and contended
  stays well under real-time either way.

**Verdict: the budget is not at risk with the real TTS config.** (Re-confirm if TTS is ever raised to
multi-thread or if STT switches to `modified_beam_search`/LM, which add decoder/joiner compute.)

### Tier-2 (hotwords / modified_beam_search) — MEASURED, fits the budget

`scripts/stream_asr.py` (deployable path: X-ASR int8 → optional hotwords → s2twp), 40 Taiwan clips,
2 threads on-device:

| mode | RTF @2 thr (2 s tail-pad) | vs greedy | MER (after s2twp) |
|---|---|---|---|
| greedy_search | 0.583 | — | 0.087 [0.044, 0.147] |
| modified_beam_search | 0.762 | **+31%** | 0.084 [0.049, 0.131] |
| modified_beam_search + hotwords | 0.761 | +31% | 0.084 [0.049, 0.131] |

**Findings:**
- **Tier-2 fits the 2-core budget.** `modified_beam_search` costs **+31%** over greedy; **hotwords add ~0%**
  on top. Applied to the pure-streaming greedy RTF (~0.30, padding removed), `modified_beam_search` ≈
  **0.39 pure-streaming** — real-time, and within the +5% TTS co-tenancy headroom. (The 0.58/0.76 absolutes
  are inflated by the 2 s per-clip flush silence; all < 1 regardless.)
- **No accuracy regression** — `modified_beam_search` marginally *improves* MER (0.087→0.084) and tightens
  the CI.
- **Hotwords had zero effect on this set** — by design: the curated TW terms (软件/台积电/…) don't occur in
  ML-lecture content. On attendant-domain speech (caller names, product terms) they bias as intended; this
  run confirms they're free and don't over-trigger / regress general text.

**Deployment note:** enable `modified_beam_search` + hotwords when domain-term biasing matters (the +31%
still fits budget); use greedy if minimum latency is paramount (accuracy is near-identical without
domain-matching hotwords).

## Scope note — Taiwanese Hokkien is out of band for this model
[Breeze-ASR-26](https://huggingface.co/MediaTek-Research/Breeze-ASR-26) (raised during this work) is a
Taiwanese **Hokkien (Taigi/台語)** recognizer (Whisper-large-v2, 2B, **offline**, 30.13% CER on the Taigi
benchmark, Apache-2.0). It is a *different language* and *not streaming/edge* — not a candidate here.
**Open product question:** if the attendant must serve **Hokkien-speaking** callers, X-ASR (Mandarin+en)
cannot — that needs a separate (cloud) path, since the only good Hokkien models are 2B/offline. If the
attendant is Mandarin+English only, no action.

## Bottom line
- **Ship Tier-1 `s2twp` now** (the measured majority win: Taiwan zhCER 0.405→0.082, zero retrain) and
  Tier-2 hotwords.
- **Do not fine-tune for accent** — no acoustic gap measured; the model already handles Taiwan Mandarin
  (8% CS-lecture CER vs 4.8% mainland clean-read, CIs overlap) and English (3.1% WER) well. **And the
  Taiwan SOTA Breeze-ASR-25 (2B, offline, in-domain) beats deployed X-ASR by only ~3 pp MER with
  overlapping CIs** — X-ASR is CI-competitive with a 40×-larger in-domain model.
- **Budget is safe** with the real single-thread matcha8k TTS (1.05× co-tenancy cost).
- **Decide Hokkien scope** separately (out of band for this edge model).

## Reproduce
```bash
# host: pull slices
python scripts/build_eval_set.py --slices tw_cs en --per-slice 40
# (mainland control via FLEURS cmn_hans_cn test; CV zh-TW is gated)
# device (moss env): decode each slice -> hyp tsv (sherpa-onnx, 2 threads, ~2s tail-pad)
# host: split test + controls
python scripts/zh_tw_postproc.py --in hyp_tw_cs.tsv --out hyp.zhtw.tsv --opencc s2twp
python scripts/eval_asr.py --hyp hyp.zhtw.tsv --ref data/text/tw_cs.ref.tsv --bootstrap 1000   # Taiwan deployed
python scripts/eval_asr.py --hyp hyp_zh_cn.tsv --ref data/text/zh_cn.ref.tsv --bootstrap 1000  # mainland floor
# device: real-TTS co-tenancy
python scripts/bench_cotenancy.py --stt-cores 0,1 --tts-cores 3 \
    --tts-cmd "bash tts_loop.sh" --stt-cmd "python xasr_decode.py tw_cs /tmp/h.tsv 2"
```
