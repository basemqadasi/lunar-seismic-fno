#!/usr/bin/env bash
set -euo pipefail
python -m lunar_fno.train.train_waveform --config configs/waveform_default.yaml
