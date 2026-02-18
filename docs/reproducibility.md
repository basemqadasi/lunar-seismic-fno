# Reproducibility

This document defines the exact source mappings and expected seed-wise metrics.

## Source Notebook Mapping

- Waveform 1D-FNO source:
  - `/home/g202210640/wrk/MQ/new_work/wav_results/FNO_MQ_waveform_v02.ipynb` (cell 33)

- Spectrogram 2D-FNO source:
  - `/home/g202210640/wrk/MQ/new_work/spec_results/FNO_MQ_spectrogram_v02.ipynb` (cell 31)

## Run Commands

```bash
python -m lunar_fno.train.train_waveform --config configs/waveform_default.yaml
python -m lunar_fno.train.train_spectrogram --config configs/spectrogram_default.yaml
```

## Frozen Seed List

- `[42, 7, 101, 2025]`

## Expected Results Artifacts

- `expected_results/waveform_seed_metrics.csv`
- `expected_results/spectrogram_seed_metrics.csv`
- `expected_results/early_stopping_summary.csv`

The values in these files are copied from the source notebook outputs and should be treated as paper-reference targets.
