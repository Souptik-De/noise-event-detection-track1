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
