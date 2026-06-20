# Evaluation — MER, code-switch boundary, and the 2-core RTF gate

Three things must be measured every time, because a win on one can be a loss on another: **accuracy**
(per language), **code-switch correctness**, and **the real-time budget at 2 threads**. A quality gain
that breaks the budget is not a gain for the attendant.

> **Prerequisite — the eval corpus does not exist yet (blocks everything).** `data/text/eval.tsv` is a
> ~10-utterance *seed* and `data/` has **no audio**. The gating "orthographic-vs-acoustic split test"
> below — which decides whether *any* GPU time is spent — is currently **unrunnable**. **First build a
> ≥70-utt-per-slice paired audio+ref dev set** (TW-CS = NTUML2021 test; English = LibriSpeech test-clean;
> mainland-zh = AISHELL-1 test; read TW = Common Voice zh-TW). See `RESEARCH.md` → "next actions".
>
> **Known limitations of the current MER metric (`scripts/eval_asr.py`) — validate before trusting it:**
> it drops **punctuation** and lowercases **casing** before aligning (so the model's punctuation/casing —
> which the downstream LLM parses — is *invisible* to the gate); it weights a 1-char CJK error equal to a
> whole English word; and the per-language slices re-align after filtering, losing cross-boundary edits.
> Add a **punctuation-F1 + casing check** as separate retention metrics, define **ε** and a bootstrap CI
> for the gate, and score **acoustic robustness** (noisy/far-field slice) — see `RESEARCH.md` gaps.

## 1. Accuracy — MER (mixed error rate)
zh-en code-switch can't be scored by a single WER or CER. Use **MER**: **CER on CJK spans + WER on
English (token/word) spans**, segment-aligned.

```bash
python scripts/eval_asr.py --hyp hyp.tsv --ref ref.tsv --metric mer
# also reports per-language: zh CER, en WER, and overall MER
```

- Score against **Traditional** references for the deployed (post-`s2twp`) output.
- **Isolate orthography from acoustics** — the key Tier-3-justification test: score CER twice, once on
  raw Simplified hyp vs a Simplified-normalized ref, once on `s2twp` hyp vs Traditional ref. If most of
  the error vanishes after `s2twp`, the gap was *orthographic* (Tier 1 fixed it) and **does not justify a
  fine-tune**. Residual error after `s2twp` is the genuinely *acoustic* part.

## 2. Code-switch boundary
Beyond MER, measure **boundary-F1**: did the model put the en↔zh switch in the right place? Boundary
errors (English words glued into a CJK run, or vice versa) hurt the LLM downstream even when the MER
looks acceptable. `eval_asr.py --metric boundary` reports precision/recall/F1 of detected switch points
against the reference.

## 3. Real-time budget — the hard gate
Every accuracy change is re-benchmarked **on the real Nano at the attendant's core budget**:

```bash
python scripts/bench_nano.py --model-dir <model> --int8 --threads 2 --wav <clip>
# encoder ms/chunk, end-to-end RTF, first-partial latency
taskset -c 0,1 python scripts/bench_nano.py ...   # enforce 2-core ceiling like the attendant does
```

Pass criteria for any candidate to be deployable:
- **RTF < ~0.6 at 2 threads** (headroom for Python greedy + co-tenant jitter while TTS runs).
- **First-partial latency** within the attendant's barge-in target (favors the 480 ms chunk variant).
- int8 — the deployed precision; never accept an fp32-only quality number.

## The MER gate for fine-tuning (Tier 3)
A fine-tuned checkpoint is accepted **only if**, on held-out multi-distribution dev sets:

| slice | metric | rule |
|---|---|---|
| Taiwan Mandarin | CER | **improves** |
| English | WER | ≤ baseline + ε |
| mainland zh | CER | ≤ baseline + ε |
| code-switch | MER + boundary-F1 | ≤ baseline + ε |
| Nano @2 threads | RTF | < 0.6, still real-time |

Select by this table, **never by training loss**. Use **≥70-utterance** eval slices — smaller sets are
too noisy (the `jetson-tts` lesson, carried over).
