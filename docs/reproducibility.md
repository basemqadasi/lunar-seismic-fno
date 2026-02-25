# Reproducibility Guide

This guide shows the exact workflow to reproduce training runs in this repository.

## 1) Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Optional quick validation:

```bash
./release_check.sh
```

## 2) Prepare data

Download the NPZ datasets from Figshare to a local data directory.

Figshare record:

- <https://doi.org/10.6084/m9.figshare.30209080.v1>

Expected NPZ files:

- `Combined_EQ_MQ64_data.npz`
- `EQ_event_noise_data_.npz`
- `PSE_MQ_test_data.npz`

Example layout:

```text
/home/you/data/lunar_fno/
  Combined_EQ_MQ64_data.npz
  EQ_event_noise_data_.npz
  PSE_MQ_test_data.npz
```

Then update dataset paths in:

- `configs/waveform_default.yaml`
- `configs/spectrogram_default.yaml`

Minimum path updates:

- waveform config:
  - `paths.eq_npz -> .../Combined_EQ_MQ64_data.npz`
  - `paths.mq_npz -> .../PSE_MQ_test_data.npz`
- spectrogram config:
  - `paths.eq_npz -> .../EQ_event_noise_data_.npz`
  - `paths.mq_npz -> .../PSE_MQ_test_data.npz`

Data key and shape requirements are documented in `docs/data_contract.md`.

## 3) Run training

Waveform (1D FNO):

```bash
python -m lunar_fno.train.train_waveform --config configs/waveform_default.yaml
```

Spectrogram (2D FNO):

```bash
python -m lunar_fno.train.train_spectrogram --config configs/spectrogram_default.yaml
```

## 4) Inspect outputs

Each modality writes under its `paths.output_dir`.
Main artifacts:

- `models/best_model_seed{seed}.pth`
- `models/final_model_seed{seed}.pth`
- `training_history_seed{seed}.csv`
- `metrics_eq.csv`
- `metrics_mq.csv`
- `early_stopping.csv`

## 5) Reproducibility notes

- Default seed list is `[42, 7, 101, 2025]`.
- Data split is controlled by `training.split_seed`.
- Exact reproducibility can still vary by GPU, driver, CUDA, and PyTorch version.
- Keep package versions fixed when comparing reruns.
