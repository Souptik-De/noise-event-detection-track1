# Noise Event Detection — Track 1 (IndoML 2026 Datathon)

Track 1 — Noise Event Detection: predict onset/offset timestamps of noise events
in speech clips. Scored by Event F1 + Segment Dice (max 2.0). Submissions are
class-agnostic.

> Source: filtered from
> [`nagshubhadip/noise-event-detection-and-removal`](https://github.com/nagshubhadip/noise-event-detection-and-removal)
> (`main` @ `32cff11`). Only `track1_detection/` + spec PDF retained; Track 2
> removed. Original baselines kept untouched; `modular/` splits are the same
> logic in focused modules. No upstream LICENSE file; credit retained here.

## Layout

```
.
├── track1_detection/
│   ├── track1_atst_sed_baseline.py         # top-level legacy script (v1 era)
│   ├── track1_crnn_meanteacher_baseline.py # top-level legacy script (v2 era)
│   └── versions/
│       ├── v-1_track1_atst_sed_baseline/          # ATST/AST-Frame SED (100ms tokens, Dice ~0.66)
│       │   ├── v-1_track1_atst_sed_baseline.py / .ipynb
│       │   └── modular/{config,data,model,losses,train,inference,evaluate,main}.py
│       ├── v-2_track1_crnn_meanteacher_baseline/  # CRNN + Mean-Teacher (40ms grid, uses Bronze)
│       │   ├── v-2_track1_crnn_meanteacher_baseline.py
│       │   └── modular/{config,data,model,losses,train,inference,evaluate,main}.py
│       └── v-3_indoml2026_track1/                 # WavLM SED + BiGRU, mean-teacher, MIL (recommended)
│           ├── v-3_indoml2026_track1.ipynb
│           └── modular/{config,data,model,losses,train,inference,evaluate,main}.py
├── IndoML Datathon 2026.pdf                # competition spec / scoring
└── requirements.txt
```

Each `modular/` package exposes `run()` in `main.py`.

## Data

`ARTPARK-IISc/Vaani-Noise-Event-Dataset` (streaming, `Audio(decode=False)`),
100 Hz `(8, T)` labels. Tiers: Gold (`verified_timestamps`, weight 1.0) /
Silver (`unverified_timestamps`, 0.75) / Bronze (`no_timestamps`, weak/unlabelled).
Set `HF_TOKEN` and accept dataset terms first.

## Running (Kaggle GPU recommended)

```bash
pip install -r requirements.txt
# v3 (recommended):
#   Kaggle: enable GPU + Internet, set HF_TOKEN secret, run
#   track1_detection/versions/v-3_indoml2026_track1/v-3_indoml2026_track1.ipynb
#   top-to-bottom, or:
python -m track1_detection.versions.v-3_indoml2026_track1.modular.main
# v1 / v2 scripts:
python track1_detection/versions/v-1_track1_atst_sed_baseline/v-1_track1_atst_sed_baseline.py
python track1_detection/versions/v-2_track1_crnn_meanteacher_baseline/v-2_track1_crnn_meanteacher_baseline.py
```

Outputs: `predictions.jsonl` + `submission_track1.zip` (JSONL at ZIP root, one
line per clip, empty `events: []` when nothing detected).

## Start here for Track 1 work

Use `v-3_indoml2026_track1` as the base; compare against v1/v2 for ablations.
See each version `README.md` and `modular/README.md` for module maps.
