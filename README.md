# jetson-stt — streaming real-time zh-TW/en STT for the Jetson Nano, on a 2-core budget

Pick, characterize, and **fine-tune** the best **streaming, real-time, code-switched Mandarin–English
(zh-TW / en) speech-to-text** that runs on a **Jetson Nano gen1** inside a **2-CPU-core budget** —
with or without the (sm_53) GPU. This is the STT half of an on-edge **smart real-time attendant**;
[`jetson-tts`](https://github.com/vieenrose/jetson-tts) is the symmetric TTS half.

## Why 2 cores (and not 4)

This repo is **one subsystem of a smart real-time attendant** running entirely at the edge except the
dialog brain:

```
  caller audio ──► [STT]  ──► partial/final text ──► [cloud LLM] ──► reply text ──► [TTS] ──► 8 kHz G.711 out
  (this repo)  ◄── on-device ──────────────────────────────────────────────► on-device (jetson-tts)
```

STT and TTS run **concurrently on the same 4× Cortex-A57**, plus VAD, audio I/O, the websocket/RTP
loop, and OS. STT therefore does **not** get the whole chip — its standing budget is **~2 cores**, so
that TTS can render the previous turn while STT listens to the next one. Every number in this repo is
reported **at 1/2/3/4 threads** so the attendant integrator can pick an operating point; the **2-thread
column is the one that matters**.

## Chosen baseline (validated on a real Nano)

**X-ASR zh-en streaming zipformer2 transducer** with punctuation
([`GilgameshWind/X-ASR-zh-en`](https://huggingface.co/GilgameshWind/X-ASR-zh-en), k2-fsa / icefall
export), run **int8 on sherpa-onnx, CPU only**. It was selected the hard way: among **every streaming
model we could get to run real-time on the Nano**, X-ASR holds the **record accuracy on zh/en
code-switching** — that bake-off is *why* it is the baseline, and why the work here is to push an
already-best model further (toward zh-TW) rather than to shop for a different one. It is a single
offline-streaming-unified model exported
at four streaming chunk sizes (160 / 480 / 960 / 1920 ms); architecture is fixed across variants
(6 stacks / 19 layers, dims 192·256·512·768·512·256, vocab 5000, 16 kHz, 80-d fbank).

Measured on-device (Tegra X1 / GM20B sm_53, L4T R32.5.1, CUDA 10.2; 960 ms variant; encoder ms/chunk —
the fair cross-engine metric; full numbers and methodology in [`docs/BASELINE.md`](docs/BASELINE.md)):

| threads | sherpa-onnx **int8** | sherpa-onnx fp32 | real-time? |
|--------:|---------------------:|-----------------:|:----------:|
| 1 | 562.9 ms | 649.5 ms | ✓ |
| **2** | **378.8 ms** | 444.7 ms | ✓ (**RTF ≈ 0.39** on a 960 ms chunk) |
| 3 | 330.3 ms | 396.4 ms | ✓ |
| 4 | 329.8 ms | 396.9 ms | ✓ |

**int8 sherpa-onnx CPU is the winner**: lossless transcription, comfortably real-time at **2 threads**,
and it **saturates at 3 threads** — so handing it a 4th core buys nothing and that core stays free for
TTS. CUDA on this model is *not runnable* on the Nano's onnxruntime 1.6.0 (see `docs/BASELINE.md`); the
ggml/CUDA offload path is the sibling repo [`x-asr-rapidspeech`](https://github.com/vieenrose/x-asr-rapidspeech)
and was slower than int8 sherpa at ≤2 cores. **So: sherpa-onnx int8, CPU, 2 threads.**

## The zh-TW question — what ships without retraining, what is data-bound

X-ASR is trained on **mainland-accent zh + en code-switch**. Making it *good for Taiwan* splits cleanly
in two — exactly mirroring `jetson-tts` (where Taiwan **readings shipped without retrain** but Taiwan
**accent was data-bound and out of scope**):

| Layer | Taiwan gap | Fix | Retrain? |
|---|---|---|---|
| **Orthography** | model emits Simplified; Taiwan writes Traditional | **OpenCC `s2twp` post-conversion** (phrase-aware) | ❌ no — ships today |
| **Vocabulary** | 軟體/软件, 滑鼠/鼠标, 計程車/出租车, 馬鈴薯/土豆… | **contextual hotword biasing** + a CN→TW term list | ❌ no — decode-time |
| **Code-switch boundaries** | en/zh split errors at switch points | `modified_beam_search` + hotwords + blank-penalty tuning | ❌ no — decode-time |
| **Acoustic accent** | Taiwan-Mandarin phonetics (reduced retroflex, no erhua, TW prosody, Taiwanese-influenced) | **fine-tune the zipformer encoder on labeled TW audio** | ✅ **yes — the real fine-tune** |

**Headline (measured on the Nano — [`docs/PHASE0_RESULTS.md`](docs/PHASE0_RESULTS.md)):** on 40 real
Taiwan code-switch clips, OpenCC `s2twp` post-processing cuts Taiwan zhCER **0.405 → 0.082** (~80%
relative, **zero retraining**). And the residual (8%) sits within CIs of the model's mainland clean-read
floor (4.8%) — **no Taiwan-accent acoustic gap was found, so a fine-tune is not justified.** The Taiwan
problem is the *written form*, fixed at decode time (`scripts/zh_tw_postproc.py`, `scripts/build_hotwords.py`).
Co-tenancy with the real single-thread matcha8k TTS costs STT only **~5%** — the 2-core budget holds.
Head-to-head, the Taiwan SOTA **Breeze-ASR-25** (2B, offline, GPU, in-domain) beats deployed X-ASR by only
**~3 pp MER with overlapping CIs** — the streaming 2-core model is CI-competitive with a 40×-larger model. The genuinely data-bound problem is **acoustic robustness to
Taiwan-accented speech**, and it carries the same regression risk `jetson-tts` hit with accent
fine-tuning: pulling the model toward TW audio can degrade English and mainland-zh / code-switch. The
plan therefore treats fine-tuning as **adapter/LoRA + low-LR + a retention set + an MER-gated checkpoint
selector**, not a naïve full fine-tune. See [`TRAINING.md`](TRAINING.md) and
[`docs/ZH_TW_PLAN.md`](docs/ZH_TW_PLAN.md).

### Gating reality: is there a trainable checkpoint? — YES (resolved by research)

**Tier 0 is resolved.** `GilgameshWind/X-ASR-zh-en` ships a trainable **`streaming_exp/pretrained.pt`
(2.56 GB)**, and the full icefall recipe (`train.py`/`finetune.py`/`export-onnx-streaming.py` +
`bpe.model`) is public at **[`github.com/Gilgamesh-J/X-ASR`](https://github.com/Gilgamesh-J/X-ASR)**.
Weight fine-tuning (Tiers 3–4) is feasible today — no re-train from scratch needed; just smoke-test that
the `.pt` loads against the recipe and matches the deployed dims first. icefall also ships **three**
in-tree fine-tune paths for this exact Zipformer2 (full, **bottleneck adapter**, **LoRA**). Full verified
detail, datasets, and the recommended plan: **[`docs/RESEARCH.md`](docs/RESEARCH.md)**. (The zero-retrain
levers above remain the first thing to ship regardless.)

## Layout

```
README.md            this file — system context, baseline, ship-vs-data-bound split
TRAINING.md          the fine-tune recipe for zh-TW/en (answers "how do we make it better?")
docs/
  RESEARCH.md        verified research synthesis — Tier-0 resolved, methods, datasets, decode levers, alternatives
  RUNBOOK.md         Phase-0 gate, step by step (fix → build eval set → split test → co-tenancy)
  PHASE0_RESULTS.md  MEASURED on the Nano: split test (s2twp removes 67% of error) + co-tenancy (budget at risk)
  BASELINE.md        X-ASR int8 sherpa-onnx CPU characterization @1/2/3/4 threads (measured)
  ENV_SETUP.md       sherpa-onnx + icefall env on host (train) and Nano (deploy)
  ZH_TW_PLAN.md      Taiwan-Mandarin adaptation, tier by tier (orthography → accent)
  DATASETS.md        zh-TW / en code-switch corpora (TAT, CommonVoice zh-TW, ASCEND, SEAME…)
  EVAL.md            MER (CER+WER) + code-switch boundary + RTF/latency methodology
scripts/
  zh_tw_postproc.py  OpenCC s2twp (maximal CJK runs) + CJK normalization (zero-retrain; --selftest)
  build_hotwords.py  CN→TW term TSV → Simplified sherpa-onnx hotwords file with boosts
  bench_nano.py      streaming RTF + encoder ms/chunk @ N threads on device
  bench_cotenancy.py STT RTF under TTS contention — the real 2-core gate (--selftest)
  eval_asr.py        MER/CER/WER + punctuation-F1 + casing + boundary, bootstrap CIs (--selftest)
  build_eval_set.py  assemble the ≥70-utt/slice eval corpus via HF streaming (--dry-run)
data/
  text/eval.tsv      seed zh-TW/en code-switch eval transcripts
  tw_terms/          CN→TW vocabulary overrides + hotword seed list
finetune/            icefall zipformer fine-tune recipe (LoRA/adapter, retention, MER gate)
ref/                 provenance pointers to upstream X-ASR + sibling repos
```

## Related work (newest → oldest)

- [`jetson-tts`](https://github.com/vieenrose/jetson-tts) — symmetric 8 kHz zh/en (+zh-TW) TTS half of the attendant
- [`x-asr-rapidspeech`](https://github.com/vieenrose/x-asr-rapidspeech) — X-ASR → GGUF / RapidSpeech.cpp (ggml CPU+CUDA, sm_53)
- [`RapidSpeech.cpp`](https://github.com/vieenrose/RapidSpeech.cpp) — ggml streaming-zipformer engine
- [`edge-speech-gpu-bench`](https://github.com/vieenrose/edge-speech-gpu-bench) — Maxwell-GPU offload benchmarking
- [`onnxruntime`](https://github.com/vieenrose/onnxruntime) / [`sherpa-onnx`](https://github.com/vieenrose/sherpa-onnx) — runtime forks
- [`streaming-zipformer-demo`](https://github.com/vieenrose/streaming-zipformer-demo) — multi-engine live mic demo (VAD + hotwords + opencc zh-TW)

## License

Apache-2.0, following the upstream **X-ASR** model and **icefall** / **sherpa-onnx** (k2-fsa). This repo
adds tooling, evaluation harness, and a fine-tuning recipe; it redistributes no model weights.
