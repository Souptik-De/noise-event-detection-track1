# Track 1 v3 runs

## v8 — fast smoke (2026-09-19, Kaggle GPU T4, kernel `souptikde/track1-v3-fast-nb`)

- Code: self-contained `kaggle/run_v3.ipynb` (8 embedded modules, byte-identical
  to `track1_detection/versions/v-3_indoml2026_track1/modular/`).
- `run(budget="fast", do_submission=False)`. `train_limit=100` tightens
  per-tier caps to 132 → 396 rows (132 gold / 132 silver / 132 bronze),
  train 100 / val 13 (gold only).
- Stage 1: 3 epochs frozen; Stage 2: 8 epochs unfrozen (enc lr 5e-5).
  ~0.1 min/epoch on GPU.
- Result (flat all epochs): **Combined 0.544 / 2.0 (Event-F1 0.148, Dice 0.396)**,
  best postproc `thr=0.7 med=21 min_dur=0.25 gate=0.7`.
- Checkpoint: `outputs/t1_wavlm.pt` (332MB, git-ignored, local only).
- Note: smoke data is tiny (100 train clips, 13 val) — F1/Dice are weak and
  flat as expected. Next: raise caps (`full` budget or larger `train_limit`)
  for real training.

## v9 — train_limit=2000 (2026-09-19, Kaggle GPU, kernel `souptikde/track1-v3-fast-nb`)

- Same self-contained notebook, cell 4 `TRAIN_LIMIT = 2000`, fast epochs (3+8).
- 6064 rows (2032 gold / 2032 silver / 2000 bronze), train 2000 / val 203.
- Data stream 5.9 min; ~1.5 min/epoch.
- Train curve moves: s1 best 0.5751 (ep01), s2 best student 0.6808 (ep02),
  end 0.6492. Teacher trails (0.5860 final).
- Postproc sweep on best ckpt: **Combined 0.818 / 2.0 (Event-F1 0.247,
  Dice 0.571)**, best `thr=0.7 med=15 min_dur=0.12 gate=0.7`.
- Checkpoint: `outputs/t1_wavlm_2000.pt` (332MB, git-ignored, local only).

## Tier 0 — no-retraining checks (2026-09-19, kernel `souptikde/track1-t0-tier0`)

- **T0.2 audit (keep):** `outputs/t1_wavlm_2000.pt` contains
  `which: student, score: 0.6808`, postproc `(0.7, 15, 0.12, 0.7)`. `train.py`
  saves whichever of student/teacher has higher val Combined and `main.py`
  reloads `ck["model"]` verbatim — no hardcoded teacher export. v2's
  "teacher at inference" convention is NOT inherited. Correct as-is.
- **T0.1 extended sweep (negative — keep Run B params):** regenerated Run B
  val posteriors from the ckpt (same seed/caps/split, 203 clips) on Kaggle
  GPU. Original grid point reproduced exactly (0.8184 / F1 0.2471 /
  Dice 0.5713). Extended grid thr ∈ {0.7…0.95} × min_dur ∈
  {0.05,0.12,0.25,0.3,0.4} (720 configs): EXT-BEST identical to original
  `(0.8184, 0.7, 15, 0.12, 0.7)`. Higher thr only trades recall — optimum
  confirmed, no change. Posteriors saved as `t0_posteriors.npz` (kernel output).
- **T0.3 submission dry-run (pass):** `export_track1` on 50 held-out train
  clips with Run B params → `dryrun_submission.zip` validated: single
  `predictions.jsonl` at zip root, 50/50 unique clip_ids, 2 silent clips with
  `events: []`, all events `{onset, offset}` with 0 ≤ onset < offset.
  Submission path works end-to-end (real test set still untested).

## Tier 1.1 — ema_decay ablation, rampup=5 (2026-09-19, kernel `souptikde/track1-t1-ema`)

One Commit, streamed once, 3 configs back-to-back on identical loaders
(train 2000 / val 203), 3+8 epochs each. Separate ckpts `t1_t11_*.pt`.

| config | ckpt val (pre-sweep) | teacher final | sweep Combined | F1 | Dice | postproc |
|---|---|---|---|---|---|---|
| Run B ref (0.999) | 0.6808 stu | 0.5860 | 0.8184 | 0.247 | 0.571 | (0.7,15,0.12,0.7) |
| base repeat (0.999) | 0.5886 stu | 0.5314 | 0.8070 | 0.224 | 0.583 | (0.7,21,0.12,0.7) |
| ema99 (0.99) | 0.6813 stu | **0.6788** | 0.8063 | 0.225 | 0.581 | (0.7,3,0.12,0.7) |
| ema995 (0.995) | 0.6624 stu | 0.6452 | 0.8106 | 0.223 | 0.588 | (0.7,5,0.25,0.7) |

- **Run-to-run variance ≈ 0.011** (base repeat 0.8070 vs Run B 0.8184, same
  data/loaders — init/shuffle noise). Deltas < ~0.015 not trustworthy.
- **Hypothesis half-confirmed:** decay 0.99 closes the teacher gap
  (0.095 → 0.003; many `saved teacher` lines) but does **not** improve val
  Combined (0.8063 vs 0.8184 ref / 0.8070 repeat — all within noise).
- **Decision: DISCARD for Combined (keep decay=0.999).** Skip rampup 8/12
  follow-up — no Combined signal to amplify; revisit only if Tier 2 changes
  consistency dynamics. Teacher-quality finding banked (useful if we ever
  switch to teacher-at-inference).
- Kept locally: `outputs/t1_t11_ema99.pt` (teacher-competitive ckpt).

## Tier 1.2 — category-auxiliary weight `cat_w` (2026-09-19, kernel `souptikde/track1-t1-catw`)

New `CFG["cat_w"]` knob (default 1.0 = old behavior) scaling channels 1..7
in `l_strong`/`l_weak`; ch0 always 1.0, `l_cons` untouched. One Commit,
streamed once, 3 configs on identical loaders (train 2000 / val 203).

| config | ckpt val (pre-sweep) | sweep Combined | F1 | Dice | postproc |
|---|---|---|---|---|---|
| Run B ref (cat_w=1) | 0.6808 | 0.8184 | 0.247 | 0.571 | (0.7,15,0.12,0.7) |
| catw10 repeat (1.0) | 0.6354 | 0.8153 | 0.231 | 0.584 | (0.7,15,0.12,0.7) |
| catw05 (0.5) | 0.5802 | 0.8092 | 0.228 | 0.581 | (0.7,15,0.12,0.7) |
| catw00 (0.0) | **0.7098** | **0.8668** | **0.280** | **0.587** | (0.7,21,0.12,0.7) |

- **PROMOTE: `cat_w=0.0` beats Run B by +0.048 Combined** (F1 +0.033, Dice
  +0.016 — far outside ~0.011 run noise). The 7 auxiliary category channels
  DILUTE the graded ch0 output; zeroing them also gives the best pre-sweep
  ckpt (0.7098) seen so far. New baseline to beat: **0.8668**.
- Note: with cat_w=0, Bronze still contributes via ch0 clip-tag weak loss.
- Kept locally: `outputs/t1_catw00_2000.pt`. Tier 2 builds on `cat_w=0.0`.

## Tier 2.1 — CNN branch (attempt 1: NaN, kernel `souptikde/track1-t2-cnn` v1)

`use_cnn=True` + `cat_w=0`, limit 2000. Result: **all losses NaN from step 0**,
F1 0.000 — hard fail, discarded as a run.
Diagnosis (local CPU repro): mel + CNN clean in fp32 AND bf16-autocast, so the
prime suspect is CUDA-fp16 overflow in the torchaudio power frontend
(power=2.0 squares STFT magnitudes; fp16 max 65504) under the train loop's
`autocast("cuda")` — v2 trained this frontend in fp32.
Fix: frontend (mel + standardize detached, CNN kept trainable) forced to fp32
via `torch.autocast(dev, enabled=False)` in `WavLMSED.forward`; verified
locally (no NaN, mel detached, CNN gets grad). Retrying below.

## Tier 2.1 — CNN branch (attempt 2: finite, NEGATIVE, kernel `track1-t2-cnn` v3)

`use_cnn=True` + `cat_w=0`, limit 2000, fp32 frontend. Losses finite, but:
ckpt val 0.5583, sweep **0.7838 (F1 0.220, Dice 0.564)** vs catw00 0.8668
(Δ=-0.083, far outside noise). Teacher flat ~0.445 all run; strong-train
loss suspiciously low (0.12–0.16 vs ~0.5+ for catw00) while val oscillates
0.44–0.56 — the random CNN+merge path overfits ch0 quickly and drowns the
pretrained WavLM signal within 11 epochs.
**Decision: DISCARD as implemented.** (A slower unfreeze schedule or
pretrained CNN could revive it, but protocol says don't iterate
indefinitely.) Checkpoint not kept (worse than baseline).
