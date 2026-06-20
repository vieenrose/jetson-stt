# finetune/ — icefall zipformer adaptation for Taiwan-accent zh-en

This directory holds the **Tier 3–4** fine-tuning recipe from [`../TRAINING.md`](../TRAINING.md). It is
intentionally empty of weights and runs nothing until **Tier 0** is satisfied — i.e. a trainable icefall
checkpoint + tokenizer + config for X-ASR exists (the HF release is ONNX-only). Build Tiers 1–2 first;
they need none of this.

## What goes here (when Tier 0 clears)
```
configs/        model + train configs mirroring the deployed streaming variant (causal, chunk, left-ctx)
adapter.py      LoRA / bottleneck-adapter wiring on the zipformer encoder (+ optional joiner)
train_ft.py     fine-tune loop: pruned-transducer loss, retention-set mixing, frequent ckpts
select_mer.py   MER-gated checkpoint selector over multi-distribution dev slices (docs/EVAL.md table)
accent_vector.py task-vector interpolation theta = theta_base + alpha*(theta_ft - theta_base) (Tier 4)
export_onnx.py  icefall streaming ONNX export + int8 quantization for the Nano
```

## Non-negotiables (carried from the jetson-tts accent post-mortem)
1. **Adapt the fewest parameters that move the accent.** LoRA/adapter on the encoder, freeze the rest.
   Full FT is Tier 4 / last resort — in jetson-tts both full FT and mis-placed LoRA collapsed content.
2. **Retention set in every batch.** English + mainland-zh + code-switch, ≥30–50% (docs/DATASETS.md).
   Without it English WER collapses — measured, not hypothetical.
3. **Select by MER on held-out multi-distribution dev sets, never by training loss.** Reject any
   checkpoint that regresses English WER or mainland CER beyond ε, even if TW CER improved.
4. **Train in the streaming regime** (causal=1, deployed chunk/left-context). Never FT the offline
   variant and hope the streaming export matches.
5. **int8 + re-bench on the real Nano at 2 threads.** A quality win that breaks the attendant's 2-core
   real-time budget is not a win. Gate with `../scripts/bench_nano.py`.

## Upstream recipe
Base on the icefall `zipformer` **streaming** recipe (pruned RNN-T). See `../docs/ENV_SETUP.md` for the
train environment and the export/quantize path back to sherpa-onnx.
