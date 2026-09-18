"""Kaggle / script entry for Track 1 v3 WavLM SED.

Usage:
    python kaggle_run_v3.py --budget fast
    python kaggle_run_v3.py --budget full --submit

HF token resolution order (never hardcode it):
  1. Kaggle Secrets `HF_TOKEN` (UserSecretsClient)
  2. env var `HF_TOKEN`
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

MODULAR = (
    Path(__file__).resolve().parent
    / "track1_detection"
    / "versions"
    / "v-3_indoml2026_track1"
    / "modular"
)
sys.path.insert(0, str(MODULAR))


def get_hf_token() -> str | None:
    try:
        from kaggle_secrets import UserSecretsClient

        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            return tok
    except Exception:
        pass
    return os.environ.get("HF_TOKEN")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", default="fast", choices=["fast", "full"])
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    tok = get_hf_token()
    if not tok:
        print("HF_TOKEN not found: add Kaggle Secret HF_TOKEN or export HF_TOKEN.")
        raise SystemExit(1)

    from main import run

    run(budget=args.budget, hf_token=tok, do_submission=args.submit)


if __name__ == "__main__":
    main()
