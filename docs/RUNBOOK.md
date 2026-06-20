# Runbook — Phase 0 (fix & measure), the gate before any GPU time

This is the executable version of `RESEARCH.md` → "Recommended sequence of work". Phase 0 is **no GPU,
no training** — it decides whether a fine-tune is even warranted. Do not skip to training; the split
test (step 4) is the gate.

## 0. Setup
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # sherpa-onnx, opencc, soundfile (decode + Tier-1)
pip install -r requirements-eval.txt      # datasets<4, librosa (eval-corpus assembly)
```

## 1. Done: OpenCC phrase layer fixed
`scripts/zh_tw_postproc.py` now converts maximal CJK runs (the char-by-char bug that disabled s2twp's
Taiwan-phrase layer is fixed). Verify:
```bash
python scripts/zh_tw_postproc.py --selftest      # PASS = phrase layer live (软件->軟體, 内存->記憶體)
```

## 2. Build the eval corpus (the prerequisite the gate needs)
The split test is unrunnable without paired audio+ref. Pull ≥70 utts/slice (streaming — fetches only
what it takes, not the whole datasets):
```bash
python scripts/build_eval_set.py --slices tw_cs cs_eval en zh_cn --per-slice 80
# Common Voice zh-TW is gated (Mozilla Data Collective, CC0): download manually, then fold in.
```
Outputs `data/audio/<slice>/*.wav` (16 kHz) + `data/text/<slice>.ref.tsv`. For Simplified-ref sources,
add `--to-traditional` so refs match post-s2twp hyp.

## 3. Decode each slice with the deployed model
```bash
# greedy baseline (matches the RTF-0.39 budget figure)
python scripts/bench_nano.py --model-dir <x-asr/960ms> --int8 --threads 2 \
    --wav <slice.wav> ...    # or a batch decode writing hyp_<slice>.tsv
# apply the Tier-1 zh-TW post-processing to the hypotheses
python scripts/zh_tw_postproc.py --in hyp_<slice>.tsv --out hyp_<slice>.zhtw.tsv --opencc s2twp
```

## 4. THE GATE — orthographic-vs-acoustic split test
Score with and without `s2twp` to separate "wrote it mainland" (orthographic, Tier-1 fixes free) from
"heard it wrong" (acoustic, the only thing a fine-tune helps):
```bash
# residual error AFTER s2twp = the genuinely acoustic part
python scripts/eval_asr.py --hyp hyp_tw_cs.zhtw.tsv --ref data/text/tw_cs.ref.tsv --bootstrap 1000
python scripts/eval_asr.py --hyp hyp_tw_cs.tsv      --ref data/text/tw_cs.ref.tsv   # pre-s2twp, for the delta
```
**Decision:** if `s2twp` + pure-CJK hotwords close most of the gap (MER drops to near the English/mainland
baseline, CIs overlap) → **STOP, no fine-tune.** Only a significant *acoustic* residual (survives s2twp)
justifies Tiers 3–4. The bootstrap CI tells you if the residual is real or noise at ≥70 utts.

## 5. THE BUDGET GATE — co-tenancy, not solo
The 2-core budget must hold while TTS runs. Measure under contention, not solo:
```bash
python scripts/bench_cotenancy.py --stt-cores 0,1 --tts-cores 2,3 \
    --stt-cmd "python scripts/bench_nano.py --model-dir <m> --int8 --threads 2 --wav <wav>"
# best fidelity: --tts-cmd "<launch the real jetson-tts engine>"
```
Pass = contended RTF stays < ~0.6 (headroom for jitter). If `modified_beam_search` (hotwords) or an LM is
added, re-run this — they cost decoder/joiner compute that competes with TTS.

## Only if steps 4–5 prove an acoustic gap within budget → Phase 2+ (`TRAINING.md` Tiers 3–4)
Smoke-test `pretrained.pt` from HF, assemble retention + TW data, PEFT-fine-tune (`--causal 1`),
MER-gated selection, export int8 with the **exact architecture CLI string** (`TRAINING.md` §Export),
re-run steps 4–5.

---

### Honest status of the harness (see `RESEARCH.md` gaps)
- `eval_asr.py` now does single-alignment per-language attribution + **punctuation-F1 + casing +
  bootstrap CIs** (`--selftest` passes). Define your **ε** for the MER gate from the CI half-width.
- The eval corpus still needs **acoustic-robustness slices** (noisy/far-field) — clean lecture/read audio
  may not predict attendant-mic behavior. Add MUSAN/RIR-augmented slices.
- `bench_cotenancy.py` synthetic load approximates TTS; use `--tts-cmd` with the real engine for the
  number you ship against.
