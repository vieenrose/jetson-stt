# Fine-tune results — X-ASR zh-en, on the GB10 + Jetson Nano (2026-06-21)

The contingency fine-tune (`TRAINING.md` Tier 3) was actually run end-to-end, to demonstrate the pipeline
and measure the effect. **It validates the training stack and confirms Phase-0's framing: fine-tuning pays
off for *domain/data* adaptation, at zero inference cost — there was no generic accent gap to begin with.**

## What was built (GB10 Grace-Blackwell host)
- **k2 built from source for CUDA 13 / Blackwell `sm_121`** (no prebuilt wheel exists). Required patching
  k2's cmake arch-detector (pin `sm_121`, downgrade spurious errors, force single-arch, strip a stray
  `compute_20` gencode CUDA 13 rejects) and supplying Python 3.12 headers locally (no sudo). Result:
  `k2-1.24.4.dev…+cuda13.0.torch2.12.1`, working on GPU.
- icefall + lhotse + the X-ASR recipe (`github.com/Gilgamesh-J/X-ASR`) wired up; exact deployed
  architecture reconstructed and **strict-loaded** from `streaming_exp/pretrained.pt`.

## Two findings that shaped everything
1. **`pretrained.pt` is the BASE *non-punctuation* model** (tokenizer `lang_5000`), not the deployed
   punctuation model (which ships ONNX-only). Verified by decoding the device's known-good features to the
   exact reference text only with `lang_5000` tokens. So this fine-tune adapts the *base* model.
2. **Features = lhotse `Fbank` (defaults already match the recipe's kaldifeat:** `high_freq=-400`,
   `dither=0`, `snip_edges=False`). Targets must be Simplified (`t2s`) and CJK-char space-separated
   (vocab stores `▁<char>`). With these correct, per-token loss validated the pipeline.

## Fine-tune #1 — encoder adaptation (the benchmarked model)
Encoder-only partial FT (147M trainable; decoder/joiner/embed frozen), 300 steps, **~40% retention**
(English + mainland mixed every batch), causal/streaming training regime, on **200 NTUML2021 train clips**.
Exported to int8 streaming ONNX (chunk 48 / left 256 = 960 ms variant) and run on the **real Jetson Nano**.

**On-device benchmark — fine-tuned vs original** (int8 streaming, 2 threads, 40-clip slices):

| metric | base (original) | FT | result |
|---|---|---|---|
| **Taiwan CS — MER** (NTUML2021, in-domain) | 0.411 [0.31, 0.52] | **0.125** [0.08, 0.18] | **−70%**, CIs disjoint |
| **Taiwan CS — zhCER** | 0.388 | **0.102** | −74% |
| **English WER — clean held-out** (LibriSpeech, *not* in FT data) | 0.064 [0.030, 0.113] | 0.061 [0.040, 0.086] | **unchanged — no forgetting** |
| **RTF @2 threads** | 0.579 | 0.583 | **unchanged — zero budget cost** |

**Reading it honestly:**
- The Taiwan gain is **in-domain adaptation** (train/eval are disjoint NTUML2021 clips but the same
  ML-lecture domain) — *not* evidence of a generic Taiwan-accent gap (Phase-0 found none vs the SOTA).
- Retention is clean on **English** (held-out, no leakage → no forgetting). The mainland number is
  leak-flattered (held-out FLEURS couldn't be pulled — HF throttling); treat mainland retention as
  *not yet cleanly measured*.
- FT changes weights only → **identical RTF**: a fine-tune does not cost the 2-core budget.
- The base (non-punc) model's absolute CER is weaker than the deployed punctuation model (e.g. base
  mainland CER ≫ the deployed model's 0.048 measured in `PHASE0_RESULTS.md`); the *relative* FT gain is
  the valid result, not the absolute base numbers.

## Fine-tune #2 — punctuation reconstruction (vocab extension)
The base model has no punctuation. Rather than switch to the deployed `bpe_punc` tokenizer (0% token-ID
overlap → full output retrain), I **extended** the `lang_5000` vocab with 7 punct tokens (`，。！？、；：`),
resizing the decoder-embedding / joiner-output / simple-loss projections (copying the 5000 trained rows),
and fine-tuned 400 steps on punctuated FLEURS. The model keeps all content knowledge and learns *where* to
insert punctuation. On **held-out** clips:

| before | after |
|---|---|
| `它让玩家可以通过电子运动和操作` | `他让玩家可以通过在空中移动设备来控子游戏中的运动和操作。` |
| `阴暗面下面的地壳薄一些高地下的地壳厚` | `阴暗面下面的地壳薄一些高地下的地壳厚一些。` |

So **punctuation is reconstructable by fine-tuning** (it's how the deployed model was made from the base).
Sentence-final `。` is learned reliably; `，！？` placement sharpens with more/varied punctuated data —
basic punctuation now, "precise" parity is a data-scale question.

## Three-model unified benchmark (base · FT · native-Traditional)
All three published models on the same 40 NTUML2021 Taiwan zh-TW/en code-switch clips, scored identically.
Two metrics, because two different problems get fixed:
- **Recognition MER/zhCER** — orthography-neutralized (both hyp & ref folded to Simplified via `t2s`) →
  pure acoustic recognition quality, fairly comparable across models that output different scripts.
- **Raw-output CER vs zh-TW ref** — the model's literal output vs the Traditional reference, i.e. *what the
  user reads with no post-processing*.

| Model | Native output | Recognition MER (95% CI) | Recognition zhCER | Raw-output CER vs zh-TW ref | On-Nano RTF @2thr |
|---|---|---|---|---|---|
| **base** (`...-base-onnx-demo`) | Simplified | 0.439 [0.332, 0.545] | 0.421 | 0.618 | 0.579 |
| **FT** (`...-ntuml2021-ft-demo`) | Simplified | **0.121** [0.078, 0.168] | **0.118** | 0.444 | 0.583 |
| **native** (`...-zh-tw-en-streaming-native-demo`) | **Traditional** | **0.121** [0.077, 0.175] | **0.118** | **0.128** | 0.583 |

Reading it:
1. **The fine-tune fixes recognition: 0.439 → 0.121 MER (3.6× error reduction); CIs disjoint** → real.
2. **`native` == `FT` on recognition** (same fine-tuned encoder; only the output vocab/script differs) — so
   the recognition gain is fully retained.
3. **`native` fixes output format: raw-output CER 0.444 → 0.128** against Traditional refs, purely by emitting
   Traditional zh-TW directly — **no `s2twp` post-step needed**. (base's 0.618 stacks Simplified↔Traditional
   orthography mismatch on top of poor recognition.)
4. **All three at RTF ≈ 0.58 on the Nano @ 2 threads (int8)** — `native` is byte-identical ONNX to `FT`
   (only `tokens.txt` differs → no compute change), so speed is unchanged. Live comparison:
   **[`Luigi/x-asr-zh-tw-en-compare`](https://huggingface.co/spaces/Luigi/x-asr-zh-tw-en-compare)**.

### Out-of-domain validation — Common Voice 17 zh-TW (N=500)
The 40-clip NTUML2021 test is *in-domain*. To check whether the gain is real or just ML-lecture
memorization, the same three models were run on **500 Common Voice 17 zh-TW clips** — everyday read
Taiwan Mandarin, a completely different corpus. Same unified scoring.

| Model | Native output | Recognition MER (95% CI) | Recognition zhCER | Raw-output CER vs zh-TW ref |
|---|---|---|---|---|
| **base** | Simplified | 0.298 [0.273, 0.324] | 0.298 | 0.497 |
| **FT** | Simplified | **0.137** [0.121, 0.153] | 0.134 | 0.386 |
| **native** | **Traditional** | **0.134** [0.119, 0.151] | 0.131 | **0.137** |

- **The FT gain generalizes:** base 0.298 → FT 0.137 MER = **−54% (2.2×)** on a different corpus, CIs
  disjoint. The shared factor with the FT data is **Taiwan Mandarin**, not the ML-lecture domain — so the
  adaptation transfers across corpora (broader than the in-domain 40-clip test alone implied).
- **`native` Traditional output is correct on real data with no post-step:** raw-output CER 0.137 ≈ its own
  recognition CER 0.131 → only ~0.6% residual (the one-to-many cases a static token relabel can't
  disambiguate). FT's raw output is 0.386 (Simplified, penalized vs Traditional refs).
- **`native` == `FT` on recognition** (0.134 vs 0.137, overlapping CIs) — confirmed again at scale.
- **Caveat:** these are gains over the *weak base non-punctuation* model, not a claim against the *deployed*
  punctuation X-ASR (absolute mainland CER ~0.048 in `PHASE0_RESULTS.md`). Reproduce: `finetune/work/eval_cv_tar.py`.

### Absolute anchor + the quality ceiling (the decisive result)
The benchmarks above are *relative to the weak base* (`pretrained.pt`, the only trainable checkpoint we have).
Running the **deployed punctuation X-ASR** (480 ms int8, what actually ships) on the same 500 CV17 zh-TW clips
sets the absolute bar — and it is far higher than any base fine-tune:

| Pipeline | Recognition MER | **Traditional-output CER** (attendant metric) |
|---|---|---|
| base + s2twp | 0.298 | 0.301 |
| FT + s2twp | 0.137 | 0.139 |
| native (direct, our FT) | 0.134 | 0.137 |
| **deployed + s2twp** | **0.064** | **0.068** [0.058, 0.079] |

- **The deployed model is ~2× better than our best fine-tune** (0.064 vs 0.134 recognition, CIs disjoint).
  `pretrained.pt` is a *weak* checkpoint; the production weights were never in reach of base-FT.
- **s2twp is essentially free orthography:** deployed recognition 0.064 → Traditional 0.068 (+0.004 only).
- **The strong float checkpoint is unobtainable:** `fintuned_with_punctuation.pt` is an LFS *pointer* with no
  object pushed (GitHub media CDN 404s; the HF mirror ships only the base `pretrained.pt` + ONNX), and int8
  ONNX is not trainable. So there is no checkpoint to fine-tune that could beat the deployed model.
- **Chunk size doesn't add accuracy:** sweeping the deployed model at 480/960/1920 ms + s2twp gives
  0.074 / 0.070 / 0.072 Traditional CER — flat within noise (longer chunks only lower compute cost, at higher
  latency). So 480 ms is the best operating point.

**Conclusion — the on-Nano zh-TW ceiling is the deployed 480 ms punct model + OpenCC s2twp at ~0.068
Traditional CER**, real-time at 2 cores. This triply confirms Phase-0's Tier-1 recommendation (ship deployed +
s2twp). Our fine-tunes remain valid *research* artifacts: they validated the GB10→Nano training stack and
showed base adaptation generalizes 2.2× out-of-domain — but they are not the production choice.
Reproduce: `finetune/work/eval_deployed_anchor.py`, `finetune/work/eval_chunk_sweep.py`.

### Native-from-strong (frozen encoder) — negative result
To test whether a *native Traditional* model could beat `deployed + s2twp` (0.071), the strong deployed model's
head was reconstructed from its float ONNX and retrained on its frozen `encoder_out`:
- **Gate 1**: torch decoder/joiner reconstructed from ONNX reproduce it numerically (decoder 1e-6, joiner 4e-5).
- **Gate 2**: lhotse-fbank + raw-ONNX encoder + torch head = sherpa output on 9/10 clips (exact char match).
- Head vocab-extended to Traditional + 47 one-to-many alternates (warm-started), trained with RNN-T loss on
  600 NTUML + 1500 CV-train clips, frozen encoder.

| Model (500 CV clips, same pipeline) | Traditional CER |
|---|---|
| deployed head + phrase-s2twp (baseline) | **0.0709** |
| native, untrained (pure relabel) | 0.0701 |
| native, best after training | 0.0758 |

**Verdict: a native model does not beat `deployed + s2twp`.** The untrained relabel native *ties* the baseline
(0.0701 vs 0.0709, within ±0.005 noise) — confirming the measured ceiling: s2twp already captures ~all the
orthography, so there is no CER headroom. Training the warm-started head *hurt* (drifted off the optimum; best
0.0758 > 0.0701 untrained) — the one-to-many alternates are too rare and s2twp too good at them to matter. The
only thing a native variant buys is architectural (Traditional output with no runtime OpenCC), at zero accuracy
gain. Reproduce: `finetune/work/native_strong_ft.py` (+ `native_precompute.py`, `enc_runner.py`).

**Shipped deliverable — native Traditional at deployed quality, no training.** Since the goal was to *match*
`deployed + s2twp` (not beat it), the clean answer is the untrained relabel: take the deployed 480 ms int8 model
**as-is** and bake `s2twp` into its `tokens.txt` (`scripts/build_native_strong.py`). On 500 CV17 zh-TW clips it
scores **0.0675 Traditional CER vs 0.0683 for deployed + s2twp** (tied), emits Traditional **directly with no
runtime OpenCC**, at the same speed (it *is* the deployed model) — and is **2× better than the prior native demo**
(weak-base relabel, 0.137). Published to
[`Luigi/x-asr-zh-tw-en-streaming-native-demo`](https://huggingface.co/Luigi/x-asr-zh-tw-en-streaming-native-demo)
and shown in the [compare Space](https://huggingface.co/spaces/Luigi/x-asr-zh-tw-en-compare) (Simplified / +s2twp
/ native, all from the one deployed model). Source: `space/`.

## Reproduce
Full recipe + helpers in `finetune/` (and the working tree on the GB10). The fine-tuned model is published
as a **demonstration** artifact (honest card) at
**[`Luigi/x-asr-zh-en-streaming-ntuml2021-ft-demo`](https://huggingface.co/Luigi/x-asr-zh-en-streaming-ntuml2021-ft-demo)**
(public). It is **not** a drop-in replacement for the deployed model — it adapts the weaker *base* model on
a small in-domain set (the card states this clearly).

## Bottom line
The whole round-trip works: **fine-tune on the GB10 → export to int8 streaming ONNX → run on the real
Nano**, with a 70% in-domain MER gain, no English forgetting, and zero RTF cost. This both validates the
training pipeline and reaffirms the Phase-0 decision (ship Tier-1/2; reserve fine-tuning for genuine
domain/data adaptation, which this demonstrates).
