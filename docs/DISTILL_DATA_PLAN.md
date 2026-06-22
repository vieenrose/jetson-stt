# Distillation data plan — how to get enough data to shrink X-ASR well

The half-size distillation POC plateaued at **0.46 CER (7× worse than the teacher)** purely because it had
~9 h of audio. This is the deep-research answer to "what data, real or synthetic, does it take to do it
right." Five parallel research sweeps (real Mandarin corpora, code-switch + English corpora, synthetic TTS,
pseudo-labeling at scale, distillation methodology) — sources cited inline below.

## The single biggest reframe: don't distill from scratch — adopt + fine-tune a pretrained small model

From-scratch distillation needs the teacher's data scale (Whisper used 680k h; icefall streaming zh models use
10–14k h). **A streaming model trained from scratch on 50 h hit ~84% WER** ([Google, arXiv:2008.05086]). The
data-efficient route is to **start from an existing pretrained small streaming zipformer and fine-tune it** on
zh-TW + code-switch data (icefall fine-tune at lr≈1/10 took a model −30% rel. WER on a new domain). Candidates,
all Apache-2.0:

| Start-from model | Size (int8) | Lang | Streaming | Quality | Source |
|---|---|---|---|---|---|
| **sherpa-onnx-streaming-zipformer-multi-zh-hans-2023-12-12** | **~73M** (enc 67M) | zh (~14k h) | yes | aishell-1 **1.91**, WS-net 8.54 | sherpa-onnx zoo |
| icefall-asr-zipformer-wenetspeech-streaming-large | ~76M | zh | yes | dev 8.0 / net 9.0 | k2-fsa HF |
| icefall-asr-zipformer-wenetspeech-streaming-**small** | **~33M** | zh | yes | (card empty — verify) | k2-fsa HF |
| sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 | ~190M | **zh-en** | yes | (no CER published) | csukuangfj HF |

**There is no public sub-40M *zh-en* streaming zipformer** — so the realistic "half size" target is the **~73M
multi-zh-hans** model (46% of X-ASR's 160M), fine-tuned to add English code-switch + zh-TW. A "third" (~33M)
exists for zh-only and will trail the teacher meaningfully. **Recommendation: fine-tune multi-zh-hans-73M (or
layer-drop-init from X-ASR) rather than distilling from scratch.**

## Data-volume targets (streaming zh-en, near-teacher = CER ~0.064)

- **~80M student: ~1,000 h to be usable, ~3,000–10,000 h to approach teacher CER.** Below ~1k h, CER ≈ doubles.
- **~33M student:** carries an intrinsic penalty; budget the same multi-thousand-hour pool and accept it won't
  fully reach 0.064.
- Dominant lever is **data/corpus diversity, not parameters** (adding 10k h GigaSpeech to 960 h LibriSpeech cut
  the *same* 70M model's test-other 7.36→5.50; multi-corpus zh drove AISHELL-1 CER ~4.3→1.9).

## Real data inventory

**Commercial-safe (CC0 / Apache / CC-BY[-SA]) — use if the attendant ships as a product:**

| Corpus | Hours | Accent | License | Notes |
|---|---|---|---|---|
| Common Voice zh-TW | ~77 valid | **Taiwan** | CC0 | only commercial-clean Taiwan audio; pull from Mozilla Data Collective (left HF Oct-2025), self-convert |
| Common Voice zh-CN / zh-HK | ~118 / ~262 valid | Mainland/HK | CC0 | general Mandarin, zero license risk |
| AISHELL-1 + AISHELL-3 + THCHS-30 | ~300 | Mainland | Apache-2.0 | clean read-speech base; `carlot/AIShell` streams as parquet |
| AISHELL-4 + AliMeeting | ~240 | Mainland | CC-BY-SA-4.0 | spontaneous/meeting acoustics |
| ASCEND | 10.6 | HK | CC-BY-SA-4.0 | zh-en **code-switch** (HK accent) |
| NTUML2021 (+ ML-lecture-long) | ~11 (+5) | **Taiwan + en CS** | permissive (verify) | tiny but on-target → reserve as **dev/eval** |

**Research/NC only (large) — fine for internal R&D, not a shipping product:**
- **FSW / NER-Trs (Formosa)** — **~3,000 h Taiwan-Mandarin broadcast**, the single biggest zh-TW lever; gated
  application via ACLCLP/NYCU, non-commercial.
- **Emilia ZH** — ~50k h, in-the-wild, **streams as WebDataset**, CC-BY-NC-4.0 + gated.
- **WenetSpeech** — 10k h, CC-BY-4.0 **NC** + gated + script-load.
- **TALCS / TAL_CSASR** — **~587 h, largest open zh-en code-switch** (Mainland, tutoring), on HF.
- **CS-Dialogue** — 104 h spontaneous zh-en CS (Mainland), CC-BY-NC-SA.

**English retention:** MLS-en (~44k h, CC-BY), People's Speech (~30k h, commercial-OK), GigaSpeech (10k h),
Common Voice en (CC0). ⚠️ **Native US English (LibriSpeech) measurably *hurt* code-switch** (accent mismatch);
prefer domain-matched / Taiwan-accented English, and **weight English well above its token share** (≈100:1 vs
Mandarin) so the student stays bilingual.

**Critical flags:** TAT / FSR-2020 / "Taiwan Tongues" are **Hokkien (台語), not Mandarin — do not use for Mandarin
acoustic training**. No-derivatives corpora (KeSpeech, MagicData-RAMC, Primewords, ST-CMDS, aidatatang) are
legally awkward for training. GigaSpeech-2 / MLS / VoxPopuli have **no Mandarin**.

## Synthetic data — the unlock for Taiwan code-switch specifically

There is **no large open Taiwan zh-en code-switch corpus**, so synthetic is essential here — and there's a
near-exact published precedent: **"Twister" (arXiv:2506.11130)** adapted Whisper using **BreezyVoice** (a
zh-TW CosyVoice-family TTS), **lowering real-data need 10×** and cutting **CSZS-zh-en code-switch −55.9%**,
Common Voice zh-TW −19%, on a corpus that was ~5,800 h *mostly synthetic* with only 10 h real English.

**Recipe (evidence-backed):**
- **TTS tools (Apache-2.0):** **CosyVoice2-0.5B** (proven base for the zh-TW/CS results; cross-lingual cloning)
  and **Spark-TTS-0.5B** (controllable novel-voice generation → hundreds of synthetic speakers). **GPT-SoVITS**
  (MIT) or **BreezyVoice** for authentic zh-TW accent from a ~1-min Taiwan reference clip. Seed a few real zh-TW
  reference voices via Edge-TTS (only source of native zh-TW voices) then clone.
- **Code-switch TEXT:** LLM-generate as a "Taiwanese bilingual speaker," inserting English at natural switch
  points (nouns / named entities / technical terms), few-shot from SEAME/ASCEND exemplars; inject the
  attendant's domain English terms; filter with an LLM-as-judge for naturalness (not perplexity).
- **Mix:** **~1:1 real:synthetic** (cap synthetic ≤~30% if the real set is small but in-domain); keep
  **≥50–100 h real** in the pot. Spend the synthetic budget on **code-switch + rare/domain terms** (documented
  30–65% rel. gains exactly there).
- **Mandatory augmentation:** noise (SNR 0–15 dB) + RIR reverb on synthetic clips; maximize **text/phonetic
  diversity** (matters more than speaker count); skip pitch perturb and high-end vocoders (Griffin-Lim is fine).
- **Guard:** hold out a **real** CS test set — if synthetic-set accuracy climbs but real CER stalls, lower the
  synthetic ratio (ASR over-fits TTS artifacts; synthetic-only ≈ doubles WER).

## Pseudo-labeling at scale — use the *stronger* teacher

The proven blueprint is **K2D (arXiv:2407.10603)**: Whisper-large-v2 pseudo-labeled ~60k h of NTU
Mandarin-English lecture audio → a **2× smaller, 5× faster streaming student that BEAT the teacher** (−16% to
−30% MER across sets), trainable in **1–2 days on one RTX 3090** (arXiv:2409.13499).

- **Label with Breeze-ASR-25, not X-ASR.** Breeze (2B, Apache-2.0, Taiwan SOTA, **native Traditional output**)
  beats Whisper on zh-TW (7.97 vs 9.84 WER) and code-switch (CSZS 13.0 vs 29.5). A stronger teacher's labels let
  a small student exceed X-ASR; native Traditional avoids the lossy Simplified→Traditional script noise. Use
  **X-ASR or Whisper-base as a cheap cross-validation filter**, not as the labeler.
- **Filter aggressively (the dominant quality lever):** two-model agreement; drop PL-vs-ref WER >10%;
  utterance-confidence <0.9; anti-hallucination heuristics (no unigram ≥3×, words/sec ∈ [1,4]); LID + Traditional
  script check. Aggressive filtering to 100 h has matched 7,500 h.
- **Iterate** 3–6 generations (hard labels + slimIPL cache or EMA teacher; soft labels collapse CTC).
- **Unlabeled-audio sources:** CC0 Common Voice (incl. unvalidated buckets); Apache AISHELL; **for R&D** FSW/NER
  (~3k h Taiwan), Emilia, WenetSpeech. **Legal:** Taiwan has **no TDM exception** and a 2025 criminal ruling
  (Lawsnote) — **do not scrape YouTube/podcasts for a shipping product**; pursue a formal Legislative-Yuan IVOD
  data request for licensed Taiwan speech instead.

## Distillation method (once data exists)

- **Primary: sequence-level KD** on filtered teacher pseudo-labels (gains scale with *unlabeled* hours, not your
  labeled set). **Add one-best-path soft-target KD** (arXiv:2110.03334: +38.5% rel. WERR with +860 h unlabeled).
- **Init:** copy/freeze the encoder or layer-drop-init from the teacher (Distil-Whisper froze the copied encoder).
- **Augment always:** speed-perturb 3× + SpecAugment (icefall defaults); MUSAN/RIR only to match the Nano's mic.
- icefall ships **MVQ-KD** (`pruned_transducer_stateless6`, `--enable-distillation`) if going the codebook route.
- If teacher is offline vs a streaming student, add **alignment-matched / two-stage KD** (arXiv:2306.15171, −19%).

## Recommended phased execution

1. **POC v2 (days):** fine-tune **multi-zh-hans-73M** on what we already have + cheap adds — NTUML + CV-zh-TW +
   **TALCS (587 h)** + **ASCEND**, all (re)labeled by **Breeze-ASR-25**, with speed-perturb + SpecAugment.
   Target: beat the 0.46 plateau decisively and gauge the curve. (This alone is ~700 h vs the failed 9 h.)
2. **Add synthetic CS (1–2 weeks):** generate ~300–500 h zh-TW/en code-switch via CosyVoice2/Spark-TTS +
   LLM-CS-text, 1:1 with real, noise/reverb aug. Re-train; hold out a real CS eval.
3. **Scale pseudo-labels (R&D track):** Breeze-label FSW/NER (~3k h Taiwan) + Emilia ZH subset, filter hard,
   iterate 3 generations → push toward near-teacher at ~73–80M.
4. **Decide third-size (~33M)** only if the budget needs it; expect a real quality gap.

## Realized experiment — small-bilingual (30.9M) relabel + X-ASR distillation attempt

A genuinely small **bilingual** zh-en streaming zipformer *does* exist (we'd missed it): `sherpa-onnx-streaming-
zipformer-small-bilingual-zh-en-2023-02-16` — **30.9M params (zipformer1), real English BPE (494 pieces, not
just letters)**, with a **trainable torch checkpoint** at `csukuangfj/k2fsa-zipformer-bilingual-zh-en-t`
(`exp/pretrained.pt`, bpe.model). Built it in icefall's `pruned_transducer_stateless7_streaming`
(nhead 4,4,4,4,4 / attention-dims 192 / encoder-dims 256 / ff 768 / vocab 6254) — **clean strict-load**.

**Deliverable (no training): relabel-only → native Traditional zh-TW + English, no OpenCC.** Applying the
`s2twp` token-relabel to its (bare-char) Chinese tokens gives, on 500 CV-zh-TW clips (streaming int8, 2 thr):
**raw-Traditional CER 0.128** (full-context 0.108), **~2.6–5× fewer FLOPs than X-ASR** (RTF 0.021 vs 0.055 on
GB10), package ~50 MB vs X-ASR's ~169 MB. A real **speed/size-vs-accuracy trade**: 0.128 (small, fast, native)
vs X-ASR's 0.068 (5× bigger). Package: `finetune/work/pkg/small_native/`.

**Distilling X-ASR's quality *into* it failed on 9h — data wall, confirmed a 4th time.** Warm-started clean at
0.108, then fine-tuning on 9h of X-ASR pseudo-labels *disrupted* the encoder (0.108→0.218) and only clawed back
to 0.158 by step 4000 — **net-negative vs the untrained 0.108**. (A tokenizer bug — the `bpe.model` is a 500-
piece *English-only* BPE; Chinese chars must map to bare tokens.txt ids — first caused a full collapse to CER
1.0; fixed, but the underlying data limit remains.) Improving recognition needs training the encoder, which on
9h overfits. To beat 0.108→0.068 needs the **full data pipeline below (1k–10k h)**; `kd3_distill.py` is ready
to scale once that data exists.

**Net:** the fast small **relabel-only** model (0.128, native, no OpenCC) ships today; closing its gap to X-ASR
is the same data-bound distillation problem.

## Realized gathering + the environment wall (2026-06-22)

Attempted to assemble a real Taiwan corpus and distill it into the small-bilingual student (warm baseline 0.108
recognition). Results:
- **kd3 (9h):** net-negative (0.108 → 0.158). **kd4 (167h, but ~150h Mainland AISHELL):** net-zero (best stayed
  0.108) — **Mainland data does NOT help the Taiwan target; accent matters, not raw hours.** **kd5 (10.6h
  Taiwan-only):** disrupts to ~0.15 then recovers (small-data) — too little to beat 0.108. **Lesson: need Taiwan
  data AT SCALE (1k–10k h).**
- **Environment wall (key blocker):** from the GB10, bulk download of Taiwan-hosted data fails —
  - **IVOD** (Legislative Yuan, `hinet.net` CDN): sustained download hangs. The built pipeline
    `scripts/ivod_gather.py` (curl HLS → VAD-window → X-ASR pseudo-label, gov speech / Art.9 / legally clean)
    **works** and banked **~6h** before the CDN stalled — proof it scales from a Taiwan network.
  - **NCHC FLUD** (~400h, OGDL open): `nchcproxy` host hangs.
  - **matbn** (~127h, HF): **gated** — needs a 1-click access request.
  - **formosaspeech** (Taiwan Mandarin, HF): streaming hangs; parquet stores audio as **external refs** (raw
    download yields ~0h). (The formospeech *org* is otherwise Hakka, not Mandarin.)
  - **YODAS**: `datasets` schema-cast bug + Mainland-mixed.
  - HF works but is slow here. **Reachable Taiwan data from this box ≈ 10.6h.**
- **To break it (requires user/infra action, not method):**
  1. **Request matbn access** on huggingface.co/datasets/formospeech/matbn (your HF account) → +127h reliable.
  2. **Run `scripts/ivod_gather.py` + the NCHC FLUD URLs from a Taiwan-network machine** (picard / a TW VPS /
     Tailscale TW exit) → IVOD tens-of-thousands h + NCHC 400h, fast. The constraint is purely network *location*
     (these are .tw ISP/academic hosts).
  Then re-run `scripts/kd5_train.py` (warm small-bilingual + Taiwan data, SpecAugment, keep-best) — the recipe is
  built and proven; only the data is missing.

## BREAKTHROUGH — YouTube is the reachable, unlimited Taiwan source (2026-06-22)

The box egresses via a **HiNet Taiwan IP (219.85.x)** — so the earlier "Taiwan hosts unreachable" was wrong;
the real blocker was the **sandbox's broken ffmpeg DNS** (ffmpeg can't resolve hosts; curl/python can). YouTube
(Google global CDN) is **reachable and fast** here. The cracking recipe (`scripts/yt_gather.py`):
- **deno** (ARM binary, direct download — no `curl|sh`) as the JS runtime → yt-dlp can extract **DASH audio-only
  (itag 140)**.
- DASH is fetched by **yt-dlp's own resolver** (not ffmpeg/m3u8) → **bypasses the ffmpeg-DNS wall**; ffmpeg only
  converts the *local* file. (Live/HLS-only videos still fail via ffmpeg — filter `!is_live`.)
- One 2.4h video → 16 kHz wav in ~1 min at ~2.2 MB/s. Then 12s-window → X-ASR pseudo-label → manifest.
- Verified: PTS 公視 news + Legislative Yuan sessions transcribe cleanly (Taiwan Mandarin).

```bash
yt-dlp --js-runtimes deno -f "140/bestaudio[ext=m4a]/bestaudio" \
  --match-filter "!is_live & duration>120 & duration<14400" \
  -x --audio-format wav --postprocessor-args "ExtractAudio:-ar 16000 -ac 1" "<url|ytsearchN:query>"
```

**This routes around the whole data wall:** unlimited Taiwan-Mandarin audio (news/talk/lecture/gov), fast, from
this box. `scripts/yt_gather.py` enumerates TW searches → downloads → segments → X-ASR-labels → `yt_manifest.tsv`
(resumable, thread-pooled). Use is recognition-only (user's call: training to recognize, not reproduce). Combine
the gathered hundreds of hours with the existing Taiwan data → kd6 (warm small-bilingual + Taiwan-at-scale) — the
real test, now unblocked.

**Bottom line:** half-size at near-teacher quality is **achievable**, but via *fine-tuning a pretrained small
streaming model on ~1k–3k h* (heavily real + synthetic-CS + Breeze-pseudo-labeled) — **not** from-scratch
distillation. Commercial-safe data caps near ~700 h (CC0 + Apache + synthetic + own licensed audio); reaching
the multi-thousand-hour regime needs the research-track corpora (FSW/NER, Emilia, WenetSpeech) or licensed
Taiwan audio. Effort: POC v2 in days; production-grade in weeks.
