# Fine-tuning X-ASR for zh-TW/en on the Jetson Nano — research findings

*Lead author report, 2026-06-20. Every load-bearing claim below was put through adversarial re-verification against primary sources; where a verdict refuted or corrected a claim, the corrected version is used and the change is flagged. "VERIFIED" = confirmed against a primary source I (or the verifier) re-fetched; "INFER" = reasoned but not directly demonstrated; "UNCERTAIN" = open and explicitly flagged.*

---

## Executive summary

- **The project's founding premise is wrong, in our favor.** The local docs say "the HF release is ONNX-only / inference-only." That is FALSE: HuggingFace `GilgameshWind/X-ASR-zh-en` ships a trainable PyTorch checkpoint `streaming_exp/pretrained.pt` (~2.4 GB, resolves HTTP 200), and the full icefall recipe — `finetune.py`, `train.py`, `export-onnx-streaming.py`, plus the `bpe.model`/`bpe_punc.model` tokenizers — is public at `github.com/Gilgamesh-J/X-ASR`. **Weight fine-tuning is feasible today.** Fix README.md / TRAINING.md / finetune/README.md.
- **Ship the zero-retrain decode levers first, and fix the OpenCC bug before anything else.** `scripts/zh_tw_postproc.py` has a load-bearing bug that makes the flagship "ship-now" Taiwan localization a near-no-op (it converts CJK one char at a time, which disables s2twp's entire phrase layer). The one-line fix (convert maximal CJK *runs*) is free and is the single highest-leverage change in this report.
- **Recommended adaptation method: encoder-scoped PEFT (LoRA or bottleneck adapter, base frozen) + ~10% original-domain replay, checkpoint-selected by a multi-distribution MER gate, not by training loss.** icefall ships first-class `zipformer_lora/` and `zipformer_adapter/` recipes for this exact Zipformer2 family. PEFT folds into the linear weights at export, so it costs **zero** runtime on the Nano. Avoid naive full fine-tune — it catastrophically forgets English / mainland-zh.
- **The accent data bottleneck is real and confirmed.** No public corpus has authentic Taiwan-accent zh-en code-switch audio at scale. The only license-clean Taiwan sources are CC0 Common Voice zh-TW (~131h, now behind Mozilla Data Collective), MIT NTUML2021 (~9h test / 35.7k utts), and **synthetic / pseudo-labeled** audio (Breeze-ASR-25 as offline teacher). Retention proxies (LibriSpeech + AISHELL + ASCEND/TAL-CSASR) substitute for the proprietary original corpus.
- **No alternative streaming model beats fine-tuned X-ASR on this hardware.** Every genuinely Taiwan-accent model (Breeze-ASR-25; community zh-TW Whisper/wav2vec2) is offline; every streaming zh-en transducer is mainland-trained. NVIDIA's `parakeet-ctc-0.6b-zh-tw` *is* Taiwan-accent streaming but requires a GPU with CC≥8.0 and ≥16 GB VRAM — undeployable on a Nano gen1 (sm_53, ~4 GB). Keep X-ASR as the student.
- **Export round-trip is fully de-risked.** A fine-tuned `.pt` returns to the Nano via X-ASR's own `export-onnx-streaming.py` → ORT `quantize_dynamic` int8. ORT 1.6.0 / opset-13 compatibility verified clean. **You must pass the deployed architecture on the export CLI** (`--encoder-dim 192,256,512,768,512,256 --num-encoder-layers 2,2,4,5,4,2 --causal True`); it is NOT recovered from the checkpoint.
- **Decisive sequencing:** (0) fix OpenCC + run the orthographic-vs-acoustic split test → (1) ship decode levers → (2) only if an *acoustic* gap remains, PEFT fine-tune with retention + MER gate → (3) re-export int8 and re-bench on the Nano. Do not spend GPU time until the split test proves the residual gap is acoustic.

---

## Tier 0 — is there a trainable checkpoint?

**Verdict: SATISFIED. The "ONNX-only" premise is refuted.** Concrete next action below.

What is VERIFIED:

1. **Trainable checkpoint exists.** `GilgameshWind/X-ASR-zh-en` ships `streaming_exp/pretrained.pt` (a torch pickle; the HF blob `resolve` URL returns HTTP 200, x-linked-size ~2.4 GB) alongside the four ONNX chunk variants (160/480/960/1920 ms encoder/decoder/joiner + `tokens.txt`). Source: `https://huggingface.co/api/models/GilgameshWind/X-ASR-zh-en`, `https://huggingface.co/GilgameshWind/X-ASR-zh-en/resolve/main/streaming_exp/pretrained.pt`.

2. **Full recipe code is public.** `github.com/Gilgamesh-J/X-ASR` (linked from the HF README) carries, as real full-size blobs under `X-ASR-zh-en/zipformer/`: `finetune.py` (77 KB), `train.py`, `decode.py`, `streaming_decode.py`, `export-onnx-streaming.py`, `export.py`/`export-onnx.py`, `model.py`, `zipformer.py` (94 KB), the supporting icefall modules, and `data/lang_5000/bpe.model` + `data/lang_5000_with_punctuation/bpe_punc.model`. Verified via the recursive git-tree API. Source: `https://api.github.com/repos/Gilgamesh-J/X-ASR/git/trees/main?recursive=1`.

   **CORRECTION (verifier):** the GitHub `.pt` files are 134-byte Git-LFS pointer stubs and the GitHub LFS objects are **currently undownloadable** — the LFS batch API returns HTTP 403 "exceeded its LFS budget." **Pull the weights from HuggingFace, not GitHub LFS.** The `bpe.model`/`bpe_punc.model` files are NOT LFS-gated and download directly.

3. **Tokenizer is recoverable exactly.** Both BPE models load cleanly (`get_piece_size()=5000`) and their piece tables are byte-identical to their `tokens.txt`. The deployed punctuation model's `tokens.txt` is byte-identical to `lang_5000_with_punctuation/tokens.txt` and differs from the non-punc `lang_5000/tokens.txt` (which lacks the punctuation tokens at IDs 2–14). **Pair `bpe_punc.model` with the punctuation checkpoint; pair `bpe.model` with the base checkpoint.** Source: GitHub raw `data/lang_*` blobs.

4. **Tokenizer composition (VERIFIED, with a corrected breakdown that sums to 5000):** 3 specials (`<blk> 0`, `<sos/eos> 1`, `<unk> 4015` mid-vocab) + 4000 single-CJK tokens (SentencePiece meta-space `▁`/U+2581 prefixed) + 721 word-initial English BPE pieces + 256 Latin continuation pieces + **20 punctuation/symbol tokens** (14 meta-space-prefixed + 6 bare). The prior breakdown summed to 4980 and silently dropped the 20 punctuation tokens. The vocab is **Simplified Chinese** — systematically verified (42/42 high-frequency simplified forms present, their traditional equivalents absent), which is exactly why the Tier-1 OpenCC s2twp post-conversion is mandatory.

5. **Lineage (VERIFIED):** the modern icefall **Zipformer2** streaming recipe (`egs/.../ASR/zipformer` family) from Xiaomi/k2-fsa — the bundled `zipformer.py`/`scaling.py` are byte-for-byte the upstream master files (causal/`chunk_size`/`left_context_frames`, `ChunkCausalDepthwiseConv1d`, `torch_autocast`), **not** the legacy `pruned_transducer_stateless7_streaming`. The model was trained in the streaming/causal regime (the four chunk variants are the canonical output of one dynamic-chunk causal training). *Caveat:* the local docs' attribution of `offline_streaming_unified` to arXiv 2506.14434 is an unsupported inference — the card cites no paper, and unified streaming/offline behavior follows generically from causal Zipformer2 (offline = `chunk_size=-1`).

**Concrete next action (do before any GPU planning):** `git lfs pull` the `.pt` from HF, load it against `Gilgamesh-J/X-ASR` `model.py`, and confirm (a) which checkpoint corresponds to the punctuation tokenizer, (b) that joiner/decoder dims match the deployed ONNX, and (c) the key structure (weights-only vs optimizer state — `finetune.py --do-finetune` loads weights-only and resets the optimizer, so weights-only suffices). Then read the deployed encoder ONNX metadata to lock the exact architecture string (see Export section).

UNCERTAIN: whether ModelScope (`Gilgamesh-J/X-ASR-zh-en`) hosts cleaner per-epoch or averaged checkpoints than the single HF `pretrained.pt`; license terms on any "collected" training data for redistributing a fine-tuned derivative (HF card is Apache-2.0, but confirm).

---

## Fine-tuning method

**Recommendation: continue-train the released `.pt` with encoder-scoped PEFT + ~10% original-domain replay, checkpoint-selected by a multi-distribution MER gate. Use the icefall in-tree recipes; do not hand-roll.**

### icefall ships three supported paths for this exact Zipformer2

| Path | Recipe | Trainable params | Streaming? | Runtime cost after export |
|---|---|---|---|---|
| **LoRA** | `egs/librispeech/ASR/zipformer_lora/` | only `lora_A`/`lora_B` (base frozen) | yes (`--causal`) | **zero** (folds into linear weights) |
| **Bottleneck adapter** | `egs/librispeech/ASR/zipformer_adapter/` | ~1% of model (4 adapters/encoder layer) | yes (`--causal`) | small; exactly toggleable off |
| **Supervised (partial/full) FT** | `egs/librispeech/ASR/zipformer/finetune.py` | all loaded modules | yes (`--causal`) | none (same graph) |

All three are VERIFIED present and on the same Zipformer2 architecture as X-ASR. The local docs under-state this — LoRA does **not** need hand-adding; a dedicated `zipformer_lora` recipe exists. Sources: `https://api.github.com/repos/k2-fsa/icefall/contents/egs/librispeech/ASR`, `https://k2-fsa.github.io/icefall/recipes/Finetune/`.

### LoRA mechanics (VERIFIED, with corrections)

- `scaling.py` defines `ScaledLinear_lora(nn.Linear, LoRALayer)`: `lora_A` shape `(r, in_features)`, `lora_B` shape `(out_features, r)`, `scaling = lora_alpha/r`, base `weight.requires_grad=False`, forward `= F.linear(x, weight, bias) + (dropout(x) @ lora_A.T @ lora_B.T) * scaling`. Genuine canonical low-rank adaptation.
- **`--lora-r` defaults to 0 — LoRA is a no-op unless you set `--lora-r > 0` (e.g. 8).** `--use-lora` defaults True but is insufficient alone. (CRITICAL operational gotcha.)
- **`--base-lr` default is `0.045`** (10× the plain-finetune `0.0045`), because only the tiny LoRA params move. *(A prior draft mis-cited this as 0.0045 — dropped a zero.)*
- LoRA is applied **only inside the encoder**: attention `in_proj` (query/key/pos in `RelPositionMultiheadAttentionWeights`, value in `SelfAttention`) and the feedforward `in_proj`/`out_proj`. NOT on attention `out_proj`, the convolution module, or the decoder/joiner. Exhaustive grep finds exactly four `_lora` sites, all in the encoder. This matches the "adapt the encoder, freeze decoder/embedding/joiner" intent.
- **At export, LoRA folds into the base linear (`merge_weights`)** — `ScaledLinear_lora.eval()` does `weight.data += (lora_B @ lora_A) * scaling`, and a fresh `use_lora=False` base model is loaded with the merged weights, dropping `lora_A`/`lora_B`. **CORRECTION:** `zipformer_lora/export.py` emits only a PyTorch/TorchScript checkpoint — there is **no ONNX export in that directory**. Reaching the Nano runtime is a **two-step** process: merge via `zipformer_lora/export.py` → feed the merged `pretrained.pt` to the standard `zipformer/export-onnx-streaming.py`. Net result still costs zero on the Nano.

### Bottleneck adapter (VERIFIED, with corrections)

Inserts 4 residual adapters per `Zipformer2EncoderLayer` (encoder only), base frozen. On its LibriSpeech reference base it trains 761,344 params = 1.148% of the model and is **exactly toggleable** (`--use-adapters 0` reproduces the base bit-for-bit; LibriSpeech test-clean held at 2.23). **CORRECTION:** that exact 1.148%/761,344 figure is specific to the LibriSpeech base (encoder-dim 192,256,384,512,384,256, vocab ~500). X-ASR's larger config (encoder-dim 192,256,512,768,512,256, vocab 5000) will differ — expect ~1% order of magnitude, not that exact number.

### Selective init / freezing (VERIFIED, with an important correction)

`--init-modules` is a comma-separated list of parameter-name **prefixes**; for each, weights are copied from the donor checkpoint where `k.startswith(prefix + ".")`, gated by `assert set(src_keys) == set(dst_keys)`. **CORRECTION:** the valid prefixes are the **eight** top-level submodules of `AsrModel`, not four: `encoder_embed, encoder, decoder, joiner, simple_am_proj, simple_lm_proj, ctc_output, attention_decoder`. The local docs' set `{encoder, decoder, joiner, encoder_embed}` omits the transducer's `simple_am_proj`/`simple_lm_proj` projection heads — a prefix list that excludes them leaves those layers **randomly initialized**, the opposite of the "keep pretrained weights" intent. Also: `--init-modules` only chooses which weights are *loaded*; it does **not** set `requires_grad=False`. To truly "load encoder and freeze it," combine `--init-modules` with your own freeze loop (which is what the LoRA/adapter recipes do).

### Retention against catastrophic forgetting

`--use-mux` mixes new-domain cuts with original-domain cuts via lhotse `CutSet.mux` ("mix 5% new with 95% original"). It **requires the original training cuts**. X-ASR's ~1M-h corpus is proprietary, so retention will be a **curated public surrogate** (see Data), not the literal original CutSet — gate its adequacy empirically with the MER retention test.

### PEFT vs full FT — the evidence (VERIFIED, with two framing corrections)

- Full FT catastrophically collapses untouched languages; PEFT preserves them. Yang et al. (Interspeech 2025, arXiv 2506.21576) Table 4, **Whisper-Medium** zh-en CS: LibriSpeech test-clean **2.90** baseline → full FT **13.15** vs LoRA **3.51** vs Soft-Prompt-Tuning **3.50**; Korean 15.96 → FFT **436** (collapse) vs LoRA 16.80. **CORRECTION 1:** the local docs cite baseline as 3.40 — **the real baseline is 2.90; fix this.** **CORRECTION 2:** this is **Whisper (encoder-decoder)**, not a transducer — do not cite it as transducer proof. The transducer case is supported by MAS-LoRA (arXiv 2505.20006) and "Updating Only Encoders Prevents Catastrophic Forgetting" (arXiv 2207.00216). Source: `https://www.isca-archive.org/interspeech_2025/yang25p_interspeech.pdf`.
- **Layer-scoped FT is a cheap baseline worth trying alongside LoRA.** For an RNN-T, ~91% of relative WER improvement comes from fine-tuning just the joiner + first encoder layer (Shor et al., Interspeech 2019). **CORRECTION:** RESEARCH.md cites this as "arXiv 2306.10860" — wrong; that id is Vander Eeckt & Van hamme's weight-averaging paper. **The correct id is arXiv 1907.13511.** Also: the 91% is the dysarthric figure measured vs joiner+*entire-encoder*; the more relevant **accented-speech figure is 86%** (vs full network).
- **PEFT is not magic for large/distant shifts.** Rehearsal-free vanilla LoRA degraded Mandarin Aishell-1 CER 22.20→44.68 when adapting Chinese-Whisper to Uyghur (arXiv 2408.10680). **A Taiwan-accent shift is small and same-language, so expect only mild forgetting — but verify with the MER gate, don't assume.**
- **Experience replay reduces forgetting but the "10% sweet spot" is model-dependent.** Pekarek-Rosin & Wermter (arXiv 2307.07280): 10% ER was the sweet spot **only for Whisper and only combined with layer freezing**; for the wav2vec2/CTC models they tested, 20% ER was strictly better and ER did not reach acceptable retention at all. **Treat ~10% as a starting point to tune, not a law; pair replay with parameter-restricting PEFT.**

### Task-vector / weight-interpolation (Tier-4 fallback)

Speech-FT (arXiv 2502.12672) validates base↔fine-tuned weight-space interpolation for speech *representation* models (HuBERT/wav2vec2/WavLM). It is a sound Tier-4 fallback. **The jetson-tts "task-vector over LoRA" preference does NOT transfer** — it was driven by LoRA catastrophically degrading the TTS flow-matching CFM decoder (CER 0.37→0.97), an architecture-specific failure that does not apply to an RNN-T encoder, where the ASR evidence makes LoRA the anti-forgetting winner.

**EWC/SI:** memory-free but modest. The strongest cited number (Trinh et al., arXiv 2207.07850) is **3.2% rel on the high-WER region / 1.3% overall** — soften the RESEARCH.md "~4–5% rel WERR" framing; it is unsupported. Treat EWC as a ~1–3% supplement when replay data is unavailable, not a primary lever.

---

## Data

The accent bottleneck is real: **no public corpus has authentic Taiwan-accent zh-en code-switch audio at scale.** The table below separates Taiwan-accent sources, code-switch sources (mostly mainland/SEA — for switch-point + CS retention, NOT accent), and English/mainland retention proxies.

| Dataset | Accent / code-switch | Size | License | Access | Use |
|---|---|---|---|---|---|
| **NTUML2021** (`ky552/ML2021_ASR_ST`) | **Taiwan-Mandarin + EN code-switch** | 35,692 utts (17.8k/3.0k/14.9k); **test ~9h** | **MIT** (corrects "unstated") | HF direct (~8.2 GB parquet) | **Primary TW-accent CS fine-tune + eval.** Only free authentic TW-accent CS audio |
| `andybi7676/ntuml2021_long` | Taiwan-Mandarin + EN CS, long-form | 1,099 utts, 24–30s each (~5h) | inherits MIT | HF direct | Long-form TW CS **eval** (= Breeze's ML-Lecture-2021-Long) |
| **Common Voice zh-TW** | Taiwan-Mandarin (read, no CS) | v23: 139,721 clips, **130.55h total / 79.22h validated, 2,290 spk** | **CC0** | **Mozilla Data Collective** (account/agreement; data stays CC0) | **Largest free TW-Mandarin** FT/eval; pseudo-label target |
| MATBN | Taiwan-Mandarin broadcast (no CS) | 198h (~150h transcribed) | non-commercial | ACLCLP application | TW-Mandarin FT/eval if NC acceptable |
| NER (Formosa) | Taiwan-Mandarin spontaneous (no CS) | ~3,200h project; FSR-2018 Vol1–4 = 610.2h | non-commercial | FSW/ACLCLP application | TW-Mandarin retention/eval if NC acceptable |
| **Breeze-ASR-25** (teacher) | TW-tuned, offline | — | **Apache-2.0** | HF | **Pseudo-label teacher** for TW CS distillation |
| TAL-CSASR / TALCS | mainland zh-en CS (baseline's own) | ~587h | unnamed "permissive" | HF mirror **gated (401)**; orig `ai.100tal.com/dataset` | CS retention / switch-point |
| ASCEND (`CAiRE/ASCEND`) | **mixed HK + Taiwan + mainland** zh-en CS | 10.62h / ~12.3k utts | CC-BY-SA-4.0 | HF direct | CS **eval** slice (Breeze suite) |
| CSZS-zh-en (`ky552/cszs_zh_en`) | zh-en CS (accent undocumented) | ~35.2k utts | MIT | HF direct | CS **eval** (Breeze suite) |
| CS-Dialogue (`BAAI/CS-Dialogue`) | mainland zh-en dialogue | 104h | CC-BY-**NC**-SA-4.0 | HF direct | research only (NC blocks product use) |
| DOTA-ME-CS | mainland zh-en CS (real+synthetic) | 18.54h / 34 spk | CC-BY-4.0 | arXiv release | CS augmentation |
| ASRU2019 CS | mainland intra-sentential CS | ~200–240h CS (+500h Mandarin) | challenge registration | DataTang | CS volume |
| SEAME (LDC2015S04) | Singapore/Malaysia zh-en CS | ~192h (~63h transcribed) | LDC (paid) | LDC | switch-point research; not free, not TW |
| AISHELL-1 | mainland Mandarin read | ~178h | Apache-2.0 | OpenSLR slr33 | **free mainland-zh retention** |
| AISHELL-2 | mainland Mandarin read | ~1000h | academic (application) | application | mainland-zh retention (mirror baseline mux) |
| LibriSpeech | English read | 960h | CC-BY-4.0 | OpenSLR | **English retention** |

**The data verdict.** For a *commercial* attendant, the license-clean Taiwan triad is **CC0 Common Voice zh-TW + MIT NTUML2021 + synthetic/pseudo-labeled**. The retention mux mirrors the baseline's upstream `multi_zh_en` recipe (LibriSpeech + TAL-CSASR + AISHELL-2); free proxies are LibriSpeech + AISHELL-1 + ASCEND. Corrections folded in:

- **TAT is Taiwanese HOKKIEN, not Mandarin** — wrong language, exclude. Same for `adi-gov-tw/Taiwan-Tongues-ASR` (Hokkien) and most HF "taiwanese"/Breeze-ASR-26 models (Hokkien/Hakka/Minnan).
- **GigaSpeech2 has NO Chinese** (Thai/Indonesian/Vietnamese only) — irrelevant.
- **ASCEND is mixed-accent including Taiwan** (paper: "Hong Kong, Taiwan, and various regions in Mainland China") — recorded in HK but speaker pool includes Taiwan; do **not** treat it as a clean non-Taiwan control.
- **NTUML2021 "~11h"** (Breeze's figure) is the full real CS set per inference — only the 9h test split is primary-source-confirmed; the ~11.2 GB `dataset_size` is bytes, not hours.
- **Common Voice access changed Oct 2025** to Mozilla Data Collective (account + per-dataset agreement); the data remains CC0.

**Synthetic/pseudo-label is the only scalable route.** Breeze-ASR-25 reached Taiwan SOTA (CommonVoice16-zh-TW CER 7.97) on **10,000h synthetic Chinese + 1,738h real English + only 11h real NTUML2021** — proving synthetic augmentation scales — and is a usable offline pseudo-label teacher. Source: `https://huggingface.co/MediaTek-Research/Breeze-ASR-25`.

UNCERTAIN: TAL-CSASR's exact license string (never published; HF mirror gated); whether the free proxy retention mix preserves the original distribution well enough at ~10% replay (empirical); CSZS-zh-en's accent label; reproducing Breeze's exact CommonVoice16-zh-TW eval subset and the Formosa long-form slices.

---

## Decode-time levers that ship without retraining

All available on CPU via sherpa-onnx. **Two gating prerequisites first:**

**(A) Fix the OpenCC bug — highest leverage, free.** `scripts/zh_tw_postproc.py` line 53 converts CJK **one char at a time** (`_CJK_CHAR.sub` over a single-char class), which silently disables s2twp's entire Taiwan-phrase layer AND breaks single-char disambiguation. **VERIFIED empirically** (opencc-python-reimplemented 0.1.7): 9/9 realistic segments diverge — `软件→軟件` (should be 軟體), `内存→內存` (should be 記憶體), `头发→頭發` (should be 頭髮), etc. The per-char output is functionally identical to glyph-only s2tw, making the flagship "ship-now" fix a near-no-op. **The fix: convert maximal CJK runs** (`re.compile(r'[CJK]+').sub(...)`), which reproduces whole-string s2twp output, keeps English/ASCII byte-identical, and costs ~0.05 ms/segment.

- **s2twp is the correct config** (VERIFIED): Simplified→Traditional *with* Taiwan vocabulary (軟件→軟體, 內存→記憶體, 視頻→影片). s2tw is glyph-variant only; s2t is OpenCC-standard glyph only — both leave mainland word choices intact. s2twp is genuinely necessary because the X-ASR vocab is Simplified-only.
- **Residual pitfall after the fix** (VERIFIED against OpenCC 1.3.1): surname over-conversion `余→餘` still ships (`余先生→餘先生`, `我姓余→我姓餘`) — root cause is `STCharacters.txt` mapping `余 → 餘 余` with 餘 as default. A recent whitelist PR protects some names but OOV name spans remain corrupted. **Apply a protected-proper-noun allow-list AFTER OpenCC**, and apply OpenCC on **final** segments only (not unstable partials, to avoid flicker).
- **Two known defects to fix in the deployment path:** the reference demo `ui_monitor.py` uses `OpenCC('s2t')` (glyph-only, wrong) — confirm production imports `zh_tw_postproc.py` with s2twp. Also note run-based conversion cannot reproduce whole-string output for the handful of mixed ASCII+CJK phrase dict entries (e.g. `U盘→隨身碟`); the English-isolation guarantee wins there — accept the tiny divergence.

**(B) Run the orthographic-vs-acoustic split test.** Quantify how much of the gap s2twp + hotwords close *before* committing to any fine-tune. If the decode levers close most of it, no GPU time is warranted.

### The decode levers

| Lever | Requires | CPU cost | Notes |
|---|---|---|---|
| **Hotwords** (Aho-Corasick biasing) | `modified_beam_search` | beam-search overhead | Strongest lever for proper nouns / CS phrases |
| **`blank_penalty`** | none (greedy-compatible) | ~0 | Cheap knob to cut deletions in fast CS |
| **`modified_beam_search`** | `max_active_paths` (default 4) | scales with #paths on decoder/joiner | Unlocks hotwords/LM/LODR; tune down to 2–3 for speed |
| **RNN-LM shallow fusion / LODR** | `modified_beam_search` + external RNN-LM ONNX | **most expensive** (per-token LM pass, single-thread) | No zh-TW RNN-LM ships; Python/C++ only, **NOT C-API** |
| **`rule_fsts` / HomophoneReplacer** | none | low | Output-level ITN / rewriting |
| **Endpointing** | `enable_endpoint_detection=True` | ~0 | rule2 (~0.6–1.0s) governs turn-end responsiveness |

- **Hotwords require `modified_beam_search`** (VERIFIED): the Python `from_transducer()` raises `ValueError` if a `hotwords_file` is passed with any other method. *Precision:* the guard is specific to the `hotwords_file` argument; hotwords passed via `create_stream(hotwords=...)` are not validated in Python — they simply have no effect under greedy.
- **English/mixed hotwords need `modeling_unit=cjkchar+bpe` AND a `bpe.vocab`** (VERIFIED). The X-ASR HF repo ships **no `bpe.model` and no `bpe.vocab`** — only ONNX + `tokens.txt` (+ `.pt`). `tokens.txt` (format `<token> <id>`) is NOT a valid `bpe.vocab` (sherpa's encoder needs `<piece>\t<score>`). To enable English/mixed hotwords you must either rebuild `bpe.model` from the recipe (it's on GitHub) and run `export_bpe_vocab.py`, or synthesize a `bpe.vocab` from the BPE pieces already in `tokens.txt`. **Pure-CJK hotwords (zh-TW names) tokenize via the cjkchar path and need no `bpe.vocab`** — usable immediately.
- **LM/LODR (VERIFIED, with sign correction):** the neural LM is an external **RNN-LM** ONNX; LODR additionally subtracts a source-domain bi-gram FST. **`lodr_scale` must be a small NEGATIVE value** (sherpa default −0.1; icefall tuned −0.24) — the `0.01` in the C++ struct is a placeholder, not real usage. LODR beats shallow fusion (icefall test-clean/other 2.61/6.74 vs 2.77/7.08). **Both require `modified_beam_search` and are NOT in the C-API** — only Python/C++. UNCERTAIN: whether the deployment uses the C-API (if so, LM/LODR are unreachable and only hotwords + `blank_penalty` + endpointing + `rule_fsts` are callable). On a 2-core Nano co-resident with TTS, the per-token RNN-LM pass is likely too costly without a tiny/quantized LM — treat LM/LODR as the last-resort lever. The claim that an LM biases specifically toward zh-TW orthography is plausible but **UNVERIFIED** (all primary evidence is English-only).

UNCERTAIN: actual added RTF of `modified_beam_search` vs greedy at `max_active_paths=4` on a real Nano for this model; the empirical `hotwords_score` (sweep 0.5–2.0; demo saw score=2.0 over-bias) and `blank_penalty` (0.0–2.0) that maximize zh-TW/CS accuracy without regressing English/mainland-zh.

---

## Export & int8 back to the Nano (and the 2-core RTF gate)

The round-trip is de-risked at the source level. Path: fine-tuned `.pt` → `zipformer/export-onnx-streaming.py` (opset 13) → ORT `quantize_dynamic` int8.

- **The export script is X-ASR's own**, a near-verbatim copy of an *older* icefall `export-onnx-streaming.py`. The opset (13), encoder metadata dict, int8 block, and cache-tensor names are byte-identical to upstream. **CORRECTION:** it is NOT byte-identical in *every* block — upstream master later added `--use-int32-inputs` touching the encoder state path, which X-ASR's copy lacks (functionally equivalent only because that flag defaults off). **Do NOT pass `--use-int32-inputs 1`** — the deployed model uses int64 `processed_lens`/decoder `y`, which ORT 1.6.0 and the sherpa loader expect.
- **CRITICAL — architecture is supplied by CLI args at export, NOT recovered from the `.pt`.** `main()` builds the model from `get_params()` + `vars(args)`, then loads the `.pt` with `strict=False`. The `.pt` stores **weights only**. The deployed encoder ONNX metadata reads `num_encoder_layers=2,2,4,5,4,2` (19 layers) and `encoder_dims=192,256,512,768,512,256` — both **differ from the export script's docstring example AND the `train.py` defaults**. So you **must** pass on the export CLI: `--encoder-dim 192,256,512,768,512,256 --num-encoder-layers 2,2,4,5,4,2 --num-heads 4,4,4,8,4,4 --query-head-dim 32 --value-head-dim 12 --cnn-module-kernel 31,31,15,15,15,31 --causal True --chunk-size <8|24|48|96> --left-context-frames <matching>`. A wrong arch loads silently (`strict=False`), not loudly.
- **Streaming metadata/state contract (VERIFIED against the real shipped ONNX):** `decode_chunk_len = chunk_size*2`, `T = decode_chunk_len + 13`. The 160/480/960/1920 ms variants come from chunk-size 8/24/48/96 → `decode_chunk_len` 16/48/96/192, `T` 29/61/109/205. The encoder has 117 inputs/outputs; the state contract is `(total_layers × 6) + 2` = 19×6+2 = 116 state tensors, named `cached_key_i / cached_nonlin_attn_i / cached_val1_i / cached_val2_i / cached_conv1_i / cached_conv2_i` per layer, plus `embed_states` (`[N,128,3,19]` float) and `processed_lens` (int64 `[N]`); every input has a matching `new_` output. *(Corrects "cached_val + cached_val2" → `cached_val1`/`cached_val2`.)* **Precision:** the export writes 13+ encoder metadata keys but the sherpa loader reads only 10; `joiner_dim` is written but never read; `feature` is read-with-default and absent on these models — so "the loader reads exactly the same set" is imprecise, though every key the plan depends on is correct.
- **int8 = ORT dynamic quantization, weight-only QInt8, QOperator/IntegerOps mode** (VERIFIED): inserts `MatMulInteger` (opset 10) + `DynamicQuantizeLinear` (opset 11), never QDQ. Encoder/joiner quantize `MatMul`; decoder `MatMul`+`Gather`. Weights stored int8; activations dynamically quantized to **uint8** at runtime. **Deployed mix = encoder.int8 + joiner.int8 + decoder.fp32** (decoder barely benefits).
- **ORT 1.6.0 / opset-13 compatibility: NO pitfall (VERIFIED).** ORT 1.6.0 supports opset 13 / IR 7; the model exports at opset 13; `quantize_dynamic`'s `check_opset_version` leaves an opset-13 QInt8 model at 13; the injected ops are within ceiling; and ORT 1.6.0's CPU kernels register `MatMulInteger` (10+) and `DynamicQuantizeLinear` (11+), so they actually run. The HF repo ships **no int8 files** (fp32 ONLY) — the baseline's int8 was quantized locally, exactly the path a fine-tune re-export follows.
- **The 2-core RTF gate.** Baseline int8 runs at RTF ~0.39 @2 threads on the real Nano. PEFT folds to zero runtime, so the fine-tuned int8 graph is identical in cost to the baseline — **PEFT does not threaten the budget.** sherpa-onnx is most efficient single-threaded and saturates at ~3 threads (MLAS caps parallelism); beam-search overhead lands on the decoder/joiner loop, which competes with co-tenant TTS — so manage `max_active_paths` and avoid LM/LODR.

UNCERTAIN — **the int8 "lossless" claim is softened to NEAR-lossless.** The only evidence is transcription-exactness on a single 10s code-switch clip; no fp32-vs-int8 WER/CER is published. **Re-measure int8-vs-fp32 CER/WER on the multi-slice dev set after any fine-tune** (`eval_asr.py`), and parity-check the export with `onnx_pretrained-streaming.py` greedy decode vs the deployed model.

---

## Alternatives — fine-tune X-ASR vs adopt/distill another model

**Verdict: no ready-to-adopt Taiwan-accent zh-en streaming edge model exists. Keep X-ASR as the student.**

- **The Taiwan-accent axis and the streaming axis are disjoint** (VERIFIED via HF API sweep). Every zh-TW model on HF is OFFLINE (Whisper or wav2vec2/wavlm CTC, ~80 entries); every streaming zh-en transducer (csukuangfj bilingual zipformer, k2-fsa `multi_zh_en`/`multi_zh-hans`, FunASR Paraformer-streaming) is mainland/internal-dataset-trained with no Taiwan accent — exactly the class X-ASR beat. *Correction:* the docs' "Whisper has a fundamental architectural limitation preventing streaming" overstates it — Whisper is merely not-native to streaming and *can* be adapted (arXiv 2506.12154, 2307.14743); but published zh-TW Whispers remain offline, so this yields no ready streaming model.
- **Breeze-ASR-25 (Taiwan SOTA) is Whisper-large-v2, ~1.55B params** (HF rounds to "2B"), **offline** — undeployable on 2× A57 and non-streaming. **Teacher, not deployment candidate.** (Verified via its `config.json`: d_model 1280, 32+32 layers — identical to whisper-large-v2.)
- **NVIDIA's `parakeet-ctc-0.6b-zh-tw` IS Taiwan-accent zh-en streaming** (600M, ~90h zh-TW, Traditional Chinese output, light CS) — this refutes any blanket "no Taiwan-accent streaming checkpoint exists." **BUT it ships only as a server-side NIM container requiring an NVIDIA GPU with Compute Capability ≥8.0 and ≥16 GB VRAM.** The Jetson Nano gen1 is sm_53 (CC 5.3, ~4 GB) — far below the floor. **Not a Nano drop-in.** FunASR Paraformer-streaming (mainland 60,000h) and NeMo Nemotron 3.5 ASR (600M, GPU) likewise fail the edge budget.
- **Distillation is a valid PARALLEL experiment, not a literature-proven solution for this exact setting.** Train the X-ASR student from Breeze-ASR-25 pseudo-labels on CV zh-TW + NTUML2021. **CORRECTION:** the cited arXiv 2409.13499 does NOT "directly validate" this path — it is monolingual on 6 *European* CommonVoice languages, has **zero code-switch and no Mandarin**, uses 200–1200h/language (vs the proposed ~91h), trains **from scratch with a fresh vocab** (not continue-training a fixed 5000-token student), and reports PL-trained models *underperforming* supervised baselines. The general principle (streaming transducers can train from noisy pseudo-labels) holds; the specific claim was overstated. Run distillation alongside PEFT + retention + MER gating, not instead of it.
- **The "X-ASR won the bake-off" claim is the project's own internal benchmark** — the HF card publishes no Taiwan/CS eval numbers. A reproducible head-to-head against Breeze's eval suite (ASCEND-{OVERALL/EN/ZH/MIX}, CV-zh-TW, CSZS-zh-en, ML-Lecture-2021-Long) is still owed.

---

## Risks & open questions

**Load-bearing risks:**

1. **`.pt` provenance unconfirmed** — verify which checkpoint matches the punctuation tokenizer, that dims match the deployed ONNX, and the key structure (weights-only is sufficient for `--do-finetune`) before GPU time.
2. **Retention proxy is empirical** — a free surrogate (LibriSpeech + AISHELL-1 + ASCEND/TAL-CSASR) may not preserve the proprietary ~1M-h distribution well enough at ~10% replay. Gate with the MER retention test; do not assume.
3. **int8 "lossless" is one 10s clip** — re-measure CER/WER fp32-vs-int8 on the multi-slice dev set after fine-tune.
4. **Export architecture must be supplied by hand** — wrong CLI args load silently under `strict=False`. Read the deployed ONNX metadata to lock the exact string.
5. **Optimal PEFT placement is unpinned** — encoder attn-qv LoRA vs all-attn+FFN LoRA vs 4-per-layer adapter vs Shor-style joiner+first-layer FT. No source decides for a same-language accent shift; A/B against the MER gate. The 86–91% joiner+first-layer result suggests a very cheap layer-scoped baseline is worth running.
6. **C-API gates the LM lever** — if the deployment uses the sherpa C-API (not Python/C++), LM/LODR are unreachable. Confirm.

**Citation hygiene to fix in the local docs:** baseline test-clean `3.40 → 2.90`; Shor-91% `arXiv 2306.10860 → 1907.13511` (and note 86% accented figure); relabel 2506.21576 as Soft-Prompt-Tuning (LoRA is its baseline) and as **Whisper**, not transducer, evidence; soften EWC "4–5% rel WERR" to ~3.2% on the high-WER region; LoRA `--base-lr` is 0.045 not 0.0045; `cached_val1`/`cached_val2` naming; ASCEND is mixed-accent including Taiwan.

**Other open questions:** the exact integer `--chunk-size` list mapping to the deployed variants given X-ASR's 6-stack downsampling (decode_chunk_len predicts 8/24/48/96 — verify against `.pt` args); whether `bpe.vocab` can be synthesized from `tokens.txt` alone for English/mixed hotwords; whether PEFT + replay compound vs either alone on this architecture; Breeze pseudo-label quality on TW audio (its own ASCEND-EN WER 26.64 suggests English/CS pseudo-labels need heavier filtering); ModelScope checkpoint availability; whether the production loop applies OpenCC on final segments only.

---

## Recommended sequence of work

**Phase 0 — Fix and measure (no GPU, days).**
1. Fix `scripts/zh_tw_postproc.py` to convert maximal CJK runs (`[CJK]+`); add a protected-proper-noun allow-list after OpenCC; apply on final segments only. Confirm the deployment imports s2twp (not the demo's s2t).
2. Correct the false "ONNX-only" premise and the citation-hygiene list in README/TRAINING.md/RESEARCH.md/finetune/README.md.
3. Run the **orthographic-vs-acoustic split test**: measure MER on the eval suite with s2twp + pure-CJK hotwords + tuned `blank_penalty`/`hotwords_score`/`max_active_paths`. **If the decode levers close most of the gap, STOP — no fine-tune needed.**

**Phase 1 — Ship decode levers (no retrain).** Deploy s2twp + hotwords (`modified_beam_search`, pure-CJK first; build `bpe.vocab` if English/mixed hotwords are needed) + `blank_penalty` + endpointing tuned for the attendant. Confirm the deployment API path (Python/C++ vs C-API) to know if LM/LODR are even reachable.

**Phase 2 — Prepare for fine-tune (only if Phase 0 proves a residual *acoustic* gap).** Pull `pretrained.pt` from **HF** (not GitHub LFS); load against `Gilgamesh-J/X-ASR` `model.py`; confirm provenance and dims. Read the deployed ONNX metadata to lock the export architecture string. Assemble the data: NTUML2021 (MIT) + CV zh-TW (CC0) + Breeze pseudo-labels (filtered) for the TW target; LibriSpeech + AISHELL-1 + ASCEND/TAL-CSASR for ~10% retention replay. Compute 80-d fbank CutSets tokenized by the same 5000-token vocab.

**Phase 3 — PEFT fine-tune in the streaming regime.** Start with encoder LoRA (`--use-lora 1 --lora-r 8 --base-lr 0.045 --causal 1`, chunk/left-context matching the deployed variant) + ~10% replay. Run the cheap layer-scoped FT (joiner + first encoder layer) as a comparison baseline. **Select checkpoints by a multi-distribution MER gate** (TW-CS, English, mainland-zh, retention), never by training loss.

**Phase 4 — Export, quantize, re-bench.** Merge PEFT → `export-onnx-streaming.py` with the locked architecture CLI args → ORT `quantize_dynamic` int8 (encoder+joiner int8, decoder fp32). Parity-check via `onnx_pretrained-streaming.py`. **Re-measure int8-vs-fp32 CER/WER on the dev set** and **re-bench RTF on the real Nano** (must stay <1 co-resident with TTS at ~0.39 baseline).

**Phase 5 — Fallbacks if PEFT underperforms.** Scale Breeze pseudo-label distillation; or task-vector/weight interpolation between base and fine-tuned (Tier-4); add EWC only as a ~1–3% memory-free supplement when replay data is thin.


---

## Completeness review — gaps & next actions

A final completeness critic audited the report above. These are real gaps in the *current repo + plan*, not in the research — they are the next work.

### Gaps
- EVAL DATA DOES NOT EXIST YET — the load-bearing 'orthographic-vs-acoustic split test' (Phase 0, gates whether any GPU time is spent) is UNRUNNABLE. data/text/eval.tsv holds ~10 utterances (13 lines incl. comments) and is self-labeled 'THIS IS A SEED... requires >=70 utterances'; data/ contains NO audio at all (only text/ and tw_terms/). The report repeatedly invokes 'run the split test', 'the multi-slice dev set', 'the MER gate' as if the apparatus exists. It does not. The report never states that building the eval corpus (paired audio + Traditional refs, >=70 utts per slice across TW-CS / English / mainland-zh / retention) is the actual first blocker before Phase 0 step 3 can run.
- MER METRIC METHODOLOGY UNEXAMINED — scripts/eval_asr.py is made the acceptance gate for the entire project, yet the report never inspects it. It computes ONE uniform-cost Levenshtein over a mixed stream where each CJK char and each English word counts as one token with substitution cost 1. This silently (a) weights a 1-char CJK error equal to a whole-English-word error, (b) drops ALL punctuation before alignment so the model's advertised punctuation capability is invisible to the gate, (c) lowercases English so casing (which the model emits) is unscored, and (d) the per-language CER/WER slices are computed by filtering tokens by language and re-aligning, which destroys cross-boundary substitution accounting. Whether 'MER improves' is even a stable, comparable quantity across the s2twp orthography change is not analyzed. This is a load-bearing measurement instrument with no validation.
- ACOUSTIC ROBUSTNESS / DATA AUGMENTATION ENTIRELY ABSENT — a smart attendant is far-field, noisy, reverberant, barge-in. The report has zero treatment of MUSAN/noise augmentation, RIR/reverberation, SpecAugment, speed/tempo perturbation, or gain/SNR robustness — all standard in icefall recipes and all directly relevant to whether a Taiwan-accent gain survives real-room audio. Common Voice and NTUML2021 are clean/lecture audio; a model tuned only on them may regress in the deployment acoustic condition. No mention of matching training augmentation to the attendant's mic/acoustic environment.
- VAD / ENDPOINTING TUNING TREATED AS A ONE-LINE TABLE ROW — endpointing governs turn-end responsiveness and barge-in for a live attendant, and is the single biggest perceived-latency lever, yet it gets one row. No analysis of rule1/rule2/rule3 thresholds, min-trailing-silence, the interaction between endpoint firing and OpenCC-on-final-segments-only, or how VAD co-resident with TTS playback (echo/half-duplex) behaves. The demo's VAD is named but never analyzed.
- FIRST-PARTIAL LATENCY / CHUNK-VARIANT CHOICE UNDECIDED — EVAL.md says first-partial latency 'favors the 480 ms chunk variant' and bench_nano measures it, but the report never recommends WHICH of the four chunk variants (160/480/960/1920 ms) to deploy, nor quantifies the accuracy-vs-latency tradeoff across them, nor whether fine-tuning should target one variant or all four. For a barge-in attendant this is a primary product decision and it is left open.
- CO-TENANCY WITH TTS NOT MEASURED — the entire RTF budget premise is 'STT co-resident with TTS on 2 cores', but every RTF number (baseline 0.39, the 'PEFT folds to zero so budget is safe') is measured with STT ALONE. The report never proposes measuring STT RTF / latency jitter while TTS is actually generating on the other core(s), nor addresses memory bandwidth / cache contention on the shared Tegra X1, nor thread-oversubscription when both sherpa-onnx and the TTS engine each spin up threads. 'PEFT costs zero extra' is true for the graph but says nothing about contention headroom.
- DENOMINATOR/STAT-SIGNIFICANCE OF THE MER GATE — the gate is 'TW CER improves; English/mainland within baseline+epsilon' but epsilon is never defined, and with ~70-utt slices the confidence interval on CER/WER is wide. No power analysis, no bootstrap CI, no statement of how large a TW improvement must be to be real vs how large an English regression is tolerable. The jetson-tts '70-utt' heuristic is carried over without justifying it suffices for the small effect sizes expected from a same-language accent shift.
- PUNCTUATION/CASING REGRESSION RISK UNADDRESSED — the deployed model uses the WITH-PUNCTUATION tokenizer (bpe_punc.model, 20 punctuation tokens). Fine-tuning on NTUML2021/Common Voice (whose transcripts may have different punctuation conventions, or none) could degrade punctuation/casing — which the downstream cloud LLM relies on for parsing. The report flags pairing the right tokenizer but never flags punctuation/casing as a retention dimension to measure, and the MER gate (which drops punctuation) is blind to it.
- NUMBER/ENTITY/ITN BEHAVIOR — an attendant deals with phone numbers, dates, addresses, amounts. zh-en number reading and inverse-text-normalization (rule_fsts / HomophoneReplacer get one table row) are barely touched. No analysis of how X-ASR renders digits/dates in code-switch, or whether s2twp/rule_fsts handle Taiwan-specific number/date orthography.
- LICENSE/COMMERCIAL-DERIVATIVE CHAIN FOR THE FINE-TUNED ARTIFACT — flagged once as UNCERTAIN but never resolved despite this being a COMMERCIAL product. Breeze-ASR-25 pseudo-labels are generated by an Apache-2.0 model, but the legal status of shipping a commercial ASR whose weights were adapted using teacher-distilled labels (and using NTUML2021 MIT + CV CC0) is not worked through. ASCEND is CC-BY-SA-4.0 (share-alike) — using it even for EVAL is fine, but any inclusion in a redistributed artifact triggers copyleft. This is left dangling.

### Concrete next actions
- BUILD THE EVAL CORPUS FIRST (blocks everything). Before any Phase 0 'split test': collect/transcribe paired AUDIO+ref for >=70 utts per slice: (a) TW-accent zh-en CS = NTUML2021 test split (`huggingface-cli download ky552/ML2021_ASR_ST --repo-type dataset`), (b) English = LibriSpeech test-clean subset, (c) mainland-zh = AISHELL-1 test, (d) read TW-Mandarin = Common Voice zh-TW via Mozilla Data Collective. Produce hyp via scripts/bench_nano decode + s2twp, score with scripts/eval_asr.py. Until audio exists locally (data/ has none), the split test in the report is vaporware.
- VALIDATE OR REPLACE THE MER METRIC. Decide: should punctuation and casing be scored (they are currently dropped/lowercased)? Add a punctuation-F1 and a casing check as separate retention metrics. Replace the per-language re-alignment with a tagged single-alignment that attributes each edit to zh/en, so cross-boundary substitutions are counted once. Add bootstrap 95% CI to MER/CER/WER and DEFINE epsilon for the gate (e.g. 'English WER regression < CI half-width'). Sanity-check eval_asr.py on a known hyp/ref pair before trusting any gate decision.
- RUN THE CO-TENANCY BENCHMARK, not the solo one. Reproduce the attendant's load: start the jetson-tts engine generating in a loop on cores 2-3, then `taskset -c 0,1 python scripts/bench_nano.py --int8 --threads 2 ...` and record RTF + first-partial latency UNDER contention, for all four chunk variants. This is the real gate number; the 0.39 solo figure is not.
- CHOOSE THE DEPLOYMENT CHUNK VARIANT explicitly. Bench 160/480/960/1920 ms for accuracy (MER on the CS slice) vs first-partial latency vs RTF-under-contention, and recommend one (EVAL.md hints 480 ms). If fine-tuning, decide whether to FT a single causal chunk config or dynamic-chunk; pass the matching --chunk-size/--left-context-frames at both train and export.
- ADD AUGMENTATION TO THE FINE-TUNE PLAN. Specify MUSAN noise + RIR reverb + SpecAugment + speed-perturb in the lhotse CutSet pipeline (icefall's asr_datamodule already supports these flags), and add a noisy/far-field eval slice. Otherwise a clean-audio TW gain may not survive the attendant's mic.
- SMOKE-TEST TIER 0 NOW (cheap, no GPU). `huggingface-cli download GilgameshWind/X-ASR-zh-en streaming_exp/pretrained.pt`, load it against Gilgamesh-J/X-ASR zipformer/model.py with the deployed dims (--encoder-dim 192,256,512,768,512,256 --num-encoder-layers 2,2,4,5,4,2 --causal 1), confirm strict-load of encoder/decoder/joiner and which of pretrained.pt vs fintuned_with_punctuation.pt pairs with bpe_punc.model. Read encoder-160ms.onnx metadata_props to lock the exact export arch string. This is the one residual the finetune/README itself names.
- CONFIRM THE DEPLOYMENT API SURFACE (Python/C++ vs C-API). Grep the production attendant code for the sherpa-onnx binding in use. If C-API, LM/LODR are unreachable and the report's 'last-resort LM lever' is moot — only hotwords + blank_penalty + endpointing + rule_fsts are callable. This single fact prunes a whole branch of the plan.
- FIX THE OPENCC RUN BUG AND ADD A REGRESSION TEST. Apply the `[CJK]+` maximal-run fix to scripts/zh_tw_postproc.py (currently still single-char at line 25/53), add the protected-proper-noun allow-list (esp. surname 余 -> not 餘), apply on final segments only, and add a pytest asserting whole-run s2twp output on the 9 known-divergent segments plus byte-identity of ASCII spans. Accept the U盘->U盤 mixed-script divergence as the cost of English-isolation.
- CONTACT/REQUEST: (a) email the NTUML2021 / Breeze authors (Chih-Kai Yang, Hung-yi Lee, NTU; MediaTek Research) to ask for the exact NTUML2021 long-form (ML-Lecture-2021-Long) eval split and Breeze's CommonVoice16-zh-TW subset definition for a reproducible head-to-head; (b) apply to ACLCLP for MATBN/NER if a non-commercial eval slice is acceptable; (c) open an issue on Gilgamesh-J/X-ASR asking the author to confirm pretrained.pt provenance (weights-only vs optimizer state) and to fix the GitHub LFS 403 budget; (d) check ModelScope Gilgamesh-J/X-ASR-zh-en for averaged/per-epoch checkpoints HF lacks.
- REPRODUCE THE 'X-ASR WON THE BAKE-OFF' CLAIM. Run Breeze-ASR-25 (offline teacher) and the csukuangfj bilingual streaming zipformer over the new eval corpus alongside X-ASR, on Breeze's named slices (ASCEND-{OVERALL/EN/ZH/MIX}, CV-zh-TW, CSZS-zh-en). The internal bake-off is unaudited; this is the only way to know the residual TW gap is real and worth fine-tuning for.

### Unresolved load-bearing questions
- The orthographic-vs-acoustic split test result itself — the single decision that gates whether ANY fine-tuning happens — is unknown and currently un-runnable (no audio, ~10-utt seed eval set). Every 'only if an acoustic gap remains' conditional in the report rests on a measurement that has not and cannot yet be made.
- The actual RTF and first-partial latency of the deployed model UNDER co-tenancy with a running TTS engine on the 2-core budget. The 0.39 RTF is solo; the claim 'PEFT does not threaten the budget' is unproven for the contended case, which is the only case that matters.
- Whether the deployment uses the sherpa-onnx C-API — if so, the LM/LODR lever (treated as a real, if last-resort, option throughout) does not exist. Unconfirmed.
- pretrained.pt provenance: which checkpoint matches bpe_punc.model, whether joiner/decoder dims match the deployed ONNX, and weights-only vs optimizer-state key structure. The report calls this the gating Tier-0 next action but it is still unverified (the .pt was never actually loaded).
- Breeze-ASR-25 pseudo-label QUALITY on real Taiwan CS audio — its own ASCEND-EN WER is 26.64, so English/CS pseudo-labels likely need heavy filtering; the distillation branch's viability is unmeasured.
- Whether ~10% replay with the FREE retention surrogate (LibriSpeech+AISHELL-1+ASCEND) actually preserves the proprietary ~1M-h distribution enough to keep English WER / mainland CER within epsilon — purely empirical, untested, and epsilon itself is undefined.
- int8-vs-fp32 CER/WER on the deployed model is still a single-10s-clip anecdote ('near-lossless'); the real quantization loss on a multi-slice dev set is unmeasured, so the post-fine-tune re-quantization risk is unquantified.
- Optimal PEFT placement for a same-language accent shift (encoder attn-qv LoRA vs all-attn+FFN vs 4-per-layer adapter vs Shor-style joiner+first-layer FT) — no source decides it; the report concedes this is unpinned and it remains an open A/B with no result.
- Punctuation/casing retention under fine-tuning — load-bearing for the downstream cloud LLM's parsing, invisible to the current MER gate (which strips punctuation and lowercases), and entirely unmeasured.
- Commercial-redistribution license chain for the fine-tuned derivative (teacher-distilled labels + NTUML2021 MIT + CV CC0 + ASCEND CC-BY-SA copyleft if it ever touches training) — flagged but never resolved for a product shipping to customers.

---

### Provenance
Methodology: 8 parallel research agents (web + local repos) → adversarial re-verification of every load-bearing claim → synthesis → completeness critic. **8/8 dimensions, 40 load-bearing claims verified: 22 confirmed, 18 corrected/partial, 0 refuted.** This report is the verified auto-synthesis (it supersedes the earlier hand-written draft and corrects its citation errors).
