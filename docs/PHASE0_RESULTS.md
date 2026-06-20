# Phase-0 results — measured on the real Jetson Nano (2026-06-20)

The Phase-0 gates from `RUNBOOK.md`, run end-to-end on the device (`ssh picard@picard-desktop`). These are
**real measured numbers**, not estimates. They convert the plan's open conditionals into a verdict.

## Setup
- **Device**: Jetson Nano gen1 (Tegra X1 / GM20B, sm_53, 4× Cortex-A57, ~4 GB, ~25.6 GB/s mem BW).
- **Model**: X-ASR int8 960 ms variant, already on device at `~/xasr-bench/int8-960/`
  (`encoder.int8.onnx` + `decoder.onnx` fp32 + `joiner.int8.onnx`) + `tokens.txt`.
- **Runtime**: sherpa-onnx 1.13.3 (`OnlineRecognizer.from_transducer`, greedy), in the `moss` conda env
  (py3.12, onnxruntime 1.26). *Note: faster than the BASELINE.md ORT-1.6.0 figures — different toolchain.*
- **Data**: 40 real **Taiwan-accent zh-en code-switch** clips (NTUML2021 test, MIT), pulled with
  `scripts/build_eval_set.py`; Traditional refs containing code-switch ("decoder", "encoder"). 75.4 s audio.
- Decode: `scripts/` decode driver, ~1 s silence tail-pad to flush the 960 ms streaming chunk.

## Gate 1 — orthographic-vs-acoustic split test  →  **orthography dominates; fixed for free**

zhCER on the 40 Taiwan clips, three ways (95% bootstrap CI, 1000 resamples):

| condition | zhCER | 95% CI | MER |
|---|---|---|---|
| (a) raw model output (Simplified) vs Taiwan (Traditional) ref | **0.441** | [0.38, 0.51] | 0.433 |
| (b) **deployed**: `s2twp(hyp)` vs ref | **0.145** | [0.087, 0.22] | 0.156 |
| (c) orthography-neutralized (both → Simplified) | 0.145 | [0.087, 0.22] | 0.156 |

**Findings:**
- **~67% of the raw error was pure orthography** (Simplified↔Traditional). The Tier-1 OpenCC `s2twp`
  post-processing — including the maximal-CJK-run bug fix — **removes it with zero retraining**:
  zhCER **0.441 → 0.145**, validated on real device output.
- **(b) == (c)**: `s2twp` leaves *no* residual orthographic error (deployed output is as good as a
  fully script-neutralized comparison). The post-processing does its whole job.
- The residual **0.145 acoustic floor is confounded**, not pure Taiwan accent. It includes: streaming
  tail-truncation (some utterance-final tokens dropped — a decode/endpoint artifact, observed directly),
  spontaneous-lecture disfluencies (呃/啊), and ML-jargon domain vocabulary. The genuinely
  accent-attributable share is **below 0.145**.

**Verdict:** the dominant, deployable win (Tier-1 orthography) is **confirmed on hardware, no GPU**. The
case for a fine-tune is **not yet made** — the residual is largely artifact/confound. Per the plan, do
**not** spend GPU time until the residual is isolated: add a clean **Common Voice zh-TW read-speech**
slice + a **mainland-zh control** on the same decode path, and fix the endpoint truncation. Likely Tier-1
+ hotwords (+ maybe light LoRA) suffices.

## Gate 2 — co-tenancy budget (STT under TTS contention)  →  **bandwidth-bound, not core-bound**

`scripts/bench_cotenancy.py`, STT pinned to cores 0,1; a TTS-like CPU load on cores 2,3:

| scenario | wall | RTF | real-time? |
|---|---|---|---|
| STT solo (TTS idle) | 54.2 s | **0.475** | ✓ |
| STT + co-tenant load | 159.0 s | **1.490** | ✗ **over budget** (2.94× slowdown) |

**Findings:**
- Solo, STT is comfortably real-time at 2 cores (0.475; the absolute value is inflated by the per-clip
  1 s tail-pad — the pure streaming RTF measured 0.30).
- **Under a heavy co-tenant the RTF nearly triples and goes over real-time.** Even pinned to *disjoint*
  cores, the Tegra X1's **shared memory bandwidth + L2** are the bottleneck — the int8 encoder is
  bandwidth-bound, so a bandwidth-hungry neighbor starves it. This is the contention the solo 0.39/0.475
  figures hide, and it's exactly what the research flagged as unmeasured.

**Caveats (honest):**
- The synthetic load is 2 back-to-back GEMM workers — a **pessimistic, bandwidth-saturating proxy**;
  the real jetson-tts vocoder is likely lighter. Re-run with `--tts-cmd "<real jetson-tts>"` for the true
  number before sizing the budget.
- STT and TTS are often **temporally disjoint** in the attendant (listen, then speak); peak simultaneous
  load is mainly the **barge-in** case. The practical contention is intermittent.

**Verdict:** the 2-core budget is **not automatically safe** — co-tenancy can break real-time. Mitigations
to evaluate: cap STT at the 480 ms (lower-compute) chunk variant during TTS playback, pin/limit TTS
threads, or schedule STT and TTS to avoid simultaneous peaks. **Measure with the real TTS engine next.**

## Bottom line
- **Ship Tier-1 (`s2twp`) now** — it is the measured majority of the win, zero retrain, validated on device.
- **Hold the fine-tune** — the split test does not (yet) show a fine-tune-worthy *acoustic* gap; isolate
  the residual first.
- **Fix the budget story** — the real gate is co-tenancy, and it is currently at risk; measure against the
  actual TTS engine and add a scheduling/chunk-size mitigation.

## Reproduce
```bash
# host: pull the eval slice
python scripts/build_eval_set.py --slices tw_cs --per-slice 40
# device: decode (sherpa-onnx in the moss env) -> hyp tsv
# host: split test
python scripts/zh_tw_postproc.py --in hyp_tw_cs.tsv --out hyp.zhtw.tsv --opencc s2twp
python scripts/eval_asr.py --hyp hyp.tsv       --ref data/text/tw_cs.ref.tsv --bootstrap 1000   # (a)
python scripts/eval_asr.py --hyp hyp.zhtw.tsv  --ref data/text/tw_cs.ref.tsv --bootstrap 1000   # (b)
# device: co-tenancy
python scripts/bench_cotenancy.py --stt-cores 0,1 --tts-cores 2,3 --stt-cmd "<decode cmd>"
```
