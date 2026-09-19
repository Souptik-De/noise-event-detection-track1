# Kaggle GPU run — Track 1 v3 (WavLM SED)

Notebook `track1_detection/versions/v-3_indoml2026_track1/v-3_indoml2026_track1.ipynb`
is already Kaggle-native. No Colab shim needed (`WORK=/kaggle/working`).

## 1. Upload code (private repo — pick one)

- **A. Easiest:** Kaggle > Create Notebook > File > Upload
  `v-3_indoml2026_track1.ipynb`, or paste `kaggle_run_v3.py` below.
- **B. Git clone:** generate GitHub PAT (`repo` scope), then in a Kaggle cell:
  `git clone https://<PAT>@github.com/Souptik-De/noise-event-detection-track1.git`
- **C. Dataset:** Kaggle > Datasets > New > upload `track1_detection/` folder,
  then Add Input to the notebook.

## 2. Notebook settings

- Accelerator: **GPU T4 / P100**, Internet: **ON**
- Add-ons > Secrets > Add `HF_TOKEN` (your read-only HF token, after accepting
  https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset terms)
- The notebook reads it via `UserSecretsClient().get_secret("HF_TOKEN")` —
  never paste the token into code.

## 3. Run (v3 fast smoke first)

Use **Save & Run All (Commit)** to run (`Quick Save` only saves, does not run).
`kaggle/run_v3.ipynb` is self-contained: cell 1 writes the 8 v3 modular files,
then trains with `BUDGET="fast"` (~6k/6k/2k clips, 3+8 epochs).

Dataset streams (`streaming=True`, `Audio(decode=False)`) — no manual download;
it stops once per-tier caps are met.

## 4. Outputs

- `/kaggle/working/t1_wavlm.pt` — checkpoint (+ best thr/med/min_dur/gate)
- `/kaggle/working/predictions.jsonl` + `submission_track1.zip`
  (JSONL at ZIP root, one line per clip, `events: []` when silent)
- Right panel > Output > Download.

## 5. Local notes

- Local machine has no GPU; use it only for edits + `py_compile`.
  `torch/datasets/transformers` import errors locally are expected.
- HF token is stored locally at `~/.cache/huggingface/token` (600) for
  validation only — never committed. Rotate at
  https://huggingface.co/settings/tokens if exposed.
- No MCP server required: Kaggle Secrets + `huggingface_hub` cover auth.
