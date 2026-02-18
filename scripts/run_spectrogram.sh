#!/usr/bin/env bash
set -euo pipefail
python -m lunar_fno.train.train_spectrogram --config configs/spectrogram_default.yaml
