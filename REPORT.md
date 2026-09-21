# Noise Event Detection (Track 1, IndoML 2026 Datathon) — Full Work Report

Purpose: hand this single file to another AI for evaluation and further
improvement. It covers provenance, all three baseline techniques in detail,
our code changes, the full Kaggle infrastructure journey (including every
failure and fix), both training runs with exact numbers, artifacts, and open
problems. No secrets are contained in this file.

Date: 2026-09-19. Author repo: `Souptik-De/noise-event-detection-track1`
(private). Upstream: `nagshubhadip/noise-event-detection-and-removal`.

---

## 1. Provenance, task, data, scoring

### 1.1 Task (Track 1 — Noise Event Detection)
Predict onset/offset timestamps of noise events in speech clips.
Submissions are **class-agnostic** (only "any noise" matters at scoring).
Score = **Event F1 + Segment Dice, max 2.0**.

### 1.2 Data
`ARTPARK-IISc/Vaani-Noise-Event-Dataset` (Hugging Face, gated — requires
accepting terms + `HF_TOKEN`). Streamed with `streaming=True` and
`Audio(decode=False)`; audio decoded manually from raw bytes (version-proof
against `datasets` library decoder changes). Columns include `audio`,
`annotationQuality`, `NoiseCategory`, `NoiseSubCategoryTimeStamp`
(`{start, end, category}` spans), plus metadata
(state/district/duration/language/transcript).

### 1.3 Annotation tiers and their roles in training
| Tier | Field | Frame labels | Clip tag | Loss path |
|---|---|---|---|---|
| Gold | `verified_timestamps` | trustworthy | yes | frame (w=1.0) + weak + consistency; **validation is gold-only** |
| Silver | `unverified_timestamps` | no annotator agreement | yes | frame (w=0.4) + weak + consistency |
| Bronze | `no_timestamps` | none (zeros) | yes | weak MIL + consistency only |

### 1.4 Source repo and what we kept
Filtered from `nagshubhadip/noise-event-detection-and-removal` (`main`
@ `32cff11`, no LICENSE file — credit retained, originals untouched).
Kept: all of `track1_detection/` (v1 + v2 + v3, each with original baseline
+ `modular/` split) + competition spec PDF + root README context.
Dropped: `track2_removal/` and the both-tracks notebook.

### 1.5 Local repo layout (40+ tracked files)
```
.
├── REPORT.md / README.md / RUNS.md / KAGGLE.md / requirements.txt
├── kaggle/run_v3.ipynb          # self-contained Kaggle kernel (see §4)
├── kaggle_run_v3.py             # script-kernel entry (superseded, kept)
├── IndoML Datathon 2026.pdf
├── outputs/                     # *.pt checkpoints, git-ignored, local only
└── track1_detection/
    ├── track1_atst_sed_baseline.py, track1_crnn_meanteacher_baseline.py
    └── versions/
        ├── v-1_track1_atst_sed_baseline/   # .py + .ipynb + modular/
        ├── v-2_track1_crnn_meanteacher_baseline/  # .py + modular/
        └── v-3_indoml2026_track1/           # .ipynb + modular/
```
Each `modular/` = `config, data, model, losses, train, inference, evaluate,
main` (+ README). Git history (9 commits): filter-import `6f670a5` →
Kaggle packaging `1e04774` → notebook kernel `6e12994` → path/id fixes
`5842d0b`, `eb278f1` → self-contained notebook `f7bb0ea` → smoke log
`fa5769d` → train_limit feature `87e5273` → 2000-run log `b5cecbb`.

## 2. Baseline lineage — all three techniques

### 2.1 v-1: ATST-style (AST) SED
- Backbone: patch-wise **AST** (`MIT/ast-finetuned-audioset-10-10-0.4593`,
  stand-in for the official ATST checkpoint), fine-tuned into a frame-level
  SED head.
- Data: same streaming reader; timestamp spans rasterised to a 100 Hz
  `(8, T)` grid (ch0 = any-noise, ch1–7 = categories).
- Model: AST → drop `[CLS]`/`[distill]` tokens → max-pool over frequency →
  BiGRU → framewise `strong` head + attention-pooled `clip` head.
- Loss: focal BCE + boundary-aware edge term (targets Event-F1 tolerance).
- Training: 2 stages (frozen backbone → unfreeze at lower LR), early stop on
  rising val loss. Bronze dropped (`USE_STRONG_ONLY`).
- Inference: sliding-window posteriors, median filter, valley-splitting,
  decode-param sweep (threshold/median/merge-gap/min-dur).
- **Known cap: AST's ~100 ms time tokens cannot land inside the 50–80 ms
  Event-F1 tolerance → F1 ≈ 0.25 while Dice ≈ 0.66.** This motivated v-2.

### 2.2 v-2: CRNN + Mean-Teacher (fixes v-1's two weaknesses)
1. **Boundary resolution:** parallel **CNN branch** on log-mel spectrogram at
   a **40 ms** output grid (ATST-Frame-like), fused with the AST stream
   (FiLM/merge) → BiGRU → framewise + attention heads. The CNN supplies local
   temporal detail for sharp onset/offset edges.
2. **Unlabelled data:** **Mean-Teacher** (EMA teacher) consistency loss, so
   Bronze clips (discarded by v-1) are used as unlabelled data.
- Losses: focal BCE (labelled) + MSE student↔EMA-teacher frame-probability
  consistency (all data), sigmoid ramp-up.
- Training: stage 1 frozen + head; stage 2 unfreeze + heavy mean-teacher on
  unlabelled data. **EMA teacher used at inference.**
- Eval mirrors the official scorer (micro Event-F1 + macro temporal Dice).
- Optional real ATST-Frame backbone via `USE_ATST_FRAME=1` (+ ckpt, not
  pip-installable; off by default).

### 2.3 v-3: WavLM SED + BiGRU + MIL + mean-teacher (the run baseline)
- Encoder: **`microsoft/wavlm-base-plus`**, last **2 transformer layers
  dropped** (Schmid et al., ICASSP 2025: final layers overfit pretraining).
  WavLM emits **50 Hz frames = exactly the 20 ms label grid** (no
  interpolation except a 1–2-frame safety align).
- Head: 2-layer BiGRU-256 (dropout 0.1) → `strong` frame head (sigmoid) +
  `att` attention head → softmax-over-time attention-pooled clip prob
  (**MIL**: Bronze's only gradient path).
- Audio/label grid: 16 kHz mono, 5 s windows (`N_SAMP`), 10 ms hop (100 Hz
  labels) pooled ×2 → 20 ms outputs (`FRAMES_OUT`); 8 channels
  (any-noise + 7 categories: animal, vehicle_traffic, baby_child,
  singing_music, phone_signal_alarm, appliance_machine, human_non_speech).
  Track 1 submits only channel 0; categories are training signal.
- Data (`data.py`): one streaming pass → int16 in-memory `rows` (waveform +
  100 Hz labels + clip tags); per-tier caps stop the stream early. `SEDData`:
  fixed 5 s crops (random train / head-anchored val), **tag recomputed from
  cropped labels** for timestamped clips (anti-poisoning fix), frame-validity
  mask, per-sample tier weight. `split_and_load`: val = 10% gold;
  train = leftover gold + silver + bronze (+ optional `train_limit` cap).
- Losses (`losses.py`, `loss = l_strong + l_weak + cw·l_cons`):
  `l_strong` pos-weighted BCE (`pos_weight=4.0`, fp32 — fp16 sigmoids
  saturate) over timestamped clips only (masked by `strong`, weighted gold
  1.0/silver 0.4); `l_weak` clip-tag BCE on attention-pooled prob (all
  tiers); `l_cons` frame+clip MSE student(augmented: time-masking
  `wav_augment` + **union-label mixup**, `max` not blend, p=0.5, α=0.2)
  vs clean EMA teacher (`decay=0.999` with step-capped warmup
  `min(1-1/(step+1), decay)`), sigmoid ramp-up over 5 epochs to max 2.0.
- Training (`train.py`): stage 1 encoder frozen (AdamW head lr 1e-3, 3/6
  epochs) → stage 2 unfreeze (enc 5e-5 + head 3e-4, cosine schedule, 8/16
  epochs); AMP autocast forward + fp32 losses, grad-clip 5.0; shared
  `state{best, gstep}`; best of student/teacher on val Combined saved to
  `/kaggle/working/t1_wavlm.pt` (332 MB).
- Config (`config.py`): single `BUDGET` switch; fast = caps 6k/6k/2k,
  3+8 epochs, bs 16, default limit 100; full = gold uncapped/20k/8k, 6+16
  epochs, no limit. Our addition: `train_limit` override
  (`cap = limit+32`; fast+2000 → {2032, 2032, 2000}).
- Eval (`evaluate.py`, verbatim official scorer — do not "improve"):
  tolerance `max(0.2·ref_duration, 0.05)` s on **both** onset+offset,
  global closest-first matching, micro F1; macro per-clip Dice on 10 ms
  frames where **silent-agreeing clips score 1.0** (hence the clip gate's
  value). `cache_posteriors` runs the net once; `sweep_postprocessing`
  grids 7 thr × 6 med × 3 min_dur × 4 gate = 168 configs
  (thr 0.3–0.7, med 1–21, min_dur 0.05–0.25, gate 0–0.7).
- Inference (`inference.py`): whole-clip single pass ≤120 s (mask=None,
  whole-clip norm — no train/inference mismatch), overlapping 30 s/28 s-hop
  averaged windows above; `export_track1` applies gate → threshold → median
  → merge-gap → min-dur, drops <50 ms slivers, rounds to ms.
- Orchestration (`main.py::run`): seed → budget → stream → bucket → loaders
  → train → reload best ckpt → sweep → rewrite ckpt with best
  (thr, med, min_dur, gate) → optional `predictions.jsonl` +
  `submission_track1.zip` (JSONL at ZIP root, `events: []` when silent).
  Returns `(state, best_params)`.

## 3. Our code changes (beyond the filter)
- Docs/packaging: root `README.md` (Track-1-only), `KAGGLE.md` (GPU runbook),
  `RUNS.md` (run log), `requirements.txt` (torch, datasets, transformers,
  soundfile, librosa, jiwer, numpy, scipy, tqdm, scikit-learn,
  huggingface-hub), extended `.gitignore` (work/, *.jsonl, *.zip, *.pt).
- `kaggle_run_v3.py`: script-kernel entry (argparse budget/submit, HF token
  from Kaggle Secrets → env). Superseded by the notebook (kept for reference).
- `kaggle/run_v3.ipynb`: **self-contained** 5-cell kernel — cell 1 embeds all
  8 v3 modules (~51 KB, verified byte-identical, `py_compile`d, `config`
  import-tested) and writes them at runtime; cell 2 HF login via `HF_TOKEN`
  secret; cell 3 dataset peek; cell 4 `run(budget="fast",
  train_limit=TRAIN_LIMIT)`. Validated with `nbformat` (cell ids present).
- Feature: `train_limit` parameter exposed in v3 `modular/main.py::run()`
  and forwarded to `budget_settings()`.

## 4. Infrastructure journey (Kaggle) — failures included
Kernels: `souptikde/track1-v3-fast` (script type, v1–v2, abandoned) →
`souptikde/track1-v3-fast-nb` (notebook type, v1→v9+, the runner).
Local has no GPU (`nvidia-smi` absent); `kaggle` CLI installed, auth as
`souptikde` (`~/.kaggle/kaggle.json`, 600). HF token validated
(`Souptik-De`, stored `~/.cache/huggingface/token` 600, never committed).
1. **v1 script push:** `pip install` failed with DNS (`Temporary failure in
   name resolution`) — Internet was OFF. Fix: dropped unused `jiwer`, no
   runtime pip (soundfile/librosa preinstalled in Kaggle image).
2. **v2:** failed in 33 s on missing secret (`HF_TOKEN not found`;
   `UserSecretsClient` unreachable). Fix: user attached secret via UI.
3. **UI save conflict:** API pushes raced the open editor draft →
   `ConcurrencyViolation … ExpectedSequence=7, ActualSequence=6`. Fix: added
   proper nbformat cell ids; procedure = close tab → (we push) → open fresh.
4. **Run semantics confusion (our error):** `Quick Save` only saves;
   **`Save & Run All (Commit)`** runs. Earlier "complete in minutes" versions
   never executed (empty logs, unexecuted cells) — corrected in `KAGGLE.md`.
5. **Transient commit error:** UI `Unexpected token '<', "<html><hea…" is not
   valid JSON` — Kaggle backend HTML error page; retry succeeded (run started).
6. **Script→notebook conversion loss:** script kernel opened in UI never
   executed its code (2-min COMPLETE, zero output). Fix: rebuilt as a native
   notebook kernel ( notebook push; `cuda: True` confirmed).
7. **`ModuleNotFoundError: No module named 'config'`:** `kaggle kernels push`
   does not put companion `.py` files on the import path. Fix: self-contained
   notebook (cell 1 writes the 8 modules; log proves
   `modules: [...]` + `cuda: True`). Verified byte-identical to repo.

## 5. Training runs and results (Kaggle GPU, `do_submission=False`)

### Run A — fast smoke (`train_limit=100`, kernel v8)
- 396 rows (132 / 132 / 132), train 100 / val 13 gold. Stream 0.6 min,
  ~0.1 min/epoch.
- Flat curve: every epoch `stu 0.5434 (F1 0.148 D 0.395)`; saved once (ep00).
- Sweep: **Combined 0.544/2.0 (F1 0.148, Dice 0.396)**,
  `thr=0.7 med=21 min_dur=0.25 gate=0.7`. Checkpoint `outputs/t1_wavlm.pt`.
- Diagnosis: 100 train / 13 val clips cannot move WavLM — pipeline smoke test
  only, as designed.

### Run B — `train_limit=2000` (kernel v9, ~25 min)
- 6064 rows (2032 / 2032 / 2000), train 2000 / val 203. Stream 5.9 min,
  ~1.5 min/epoch.
- Moving curve:
  `s1 ep00 0.4488 (0.054/0.395)` → `s1 ep01 0.5751 (0.118/0.457)` [saved] →
  `s1 ep02 0.4721` → `s2 ep01 0.5863` [saved] → **`s2 ep02 0.6808
  (F1 0.160 D 0.521)` [saved, best]** → ep03–07 oscillate 0.60–0.65,
  end 0.6492 (0.151/0.499); teacher trails (0.5860 final).
- Sweep on best ckpt: **Combined 0.8184/2.0 (F1 0.247, Dice 0.571)**,
  top rows all `thr=0.70 med=15 min_dur=0.12` (gate-insensitive),
  best `(0.7, 15, 0.12, 0.7)`. Checkpoint `outputs/t1_wavlm_2000.pt`.
- Delta vs smoke: **0.544 → 0.818** (F1 +0.099, Dice +0.175) — data scale was
  the binding constraint.

## 6. Artifacts and rerun pointers
- Checkpoints (local `outputs/`, git-ignored, 332 MB each): `t1_wavlm.pt`
  (smoke), `t1_wavlm_2000.pt` (best). Kaggle copies live under the kernel's
  Output tab (`/kaggle/working/t1_wavlm.pt`, rewritten with best postproc).
- Logs: full kernel logs downloadable via
  `kaggle kernels output souptikde/track1-v3-fast-nb`; epoch tables also in
  `RUNS.md`.
- Rerun: open `.../track1-v3-fast-nb` (v9+), confirm GPU ON + Internet ON +
  `HF_TOKEN` attached, **Save & Run All (Commit)**. Change scale via cell 4
  `TRAIN_LIMIT` (or `budget="full"` for 20k/8k caps, 6+16 epochs, ~7 h).
- Deps: Kaggle image + `pip install soundfile librosa huggingface_hub`
  (cell 1); `requirements.txt` for elsewhere.

## 7. Open problems — resolved vs still open (post Tier 0–2, best Combined 0.8668)

1. **Event-F1 headroom (0.280 after T1.2):** STILL OPEN, highest value.
   Tested: v1 edge term as MSE-on-diffs at edge_w=0.5 → NEGATIVE (0.7719;
   term flat at ~0.001 vs BCE ~0.15, i.e. scaled out; F1 fell 0.280→0.202).
   Tested: v2 CNN branch fusion → NEGATIVE (0.7838; random CNN overfits ch0,
   drowns WavLM signal in 11 epochs). Untested: ~100× edge weight (risky),
   focal-BCE pairing (v1's original context), pretrained/slow-unfreeze CNN,
   T2.3 onset/offset regression heads, 10 ms output grid.
2. **Teacher lags student:** PARTLY RESOLVED. ema_decay=0.99 closes the gap
   (0.095→0.003, teacher 0.679≈student 0.681) but Combined unchanged
   (0.8063 vs 0.8184 ref) — discarded for Combined; finding banked for any
   future teacher-at-inference switch. Kept decay=0.999.
3. **Noisy model selection:** STILL OPEN. Run-to-run variance measured ≈0.011
   (base repeat 0.8070 vs Run B 0.8184, identical data); val still 203 clips.
   Deltas < ~0.015 untrustworthy without second seed / larger val (T3.2).
4. **Postproc saturation:** RESOLVED (negative). 720-config extended sweep
   (thr→0.95, min_dur→0.4) finds nothing beyond (0.7, 15, 0.12, 0.7).
   Per-clip adaptive thresholds / valley-splitting untested.
5. **Scale:** OPEN, cheapest expected win. Trend 0.544 (100) → 0.818 (2000);
   full budget (20k silver/8k bronze, all gold, 6+16 epochs, ~7 h) not yet run
   — gated behind Tier 2 win per protocol; proposed with cat_w=0 folded in.
6. **Class-aware auxiliaries:** RESOLVED — they HURT. cat_w=0 beats full
   weight by +0.048 (0.8668 vs 0.8184). Promoted; Tier 2 built on it.
   (Bronze still contributes via ch0 clip-tag loss.)
7. **Submission path:** RESOLVED (dry-run). 50-clip pseudo-test ZIP validated
   (root JSONL, unique ids, silent `events: []`, schema). Real test-set run
   (`do_submission=True` + `/kaggle/input/indoml-track1-test`) still pending.
