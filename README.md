# lunar-seismic-fno

Companion code for Moonquake/Earthquake detection with Fourier Neural Operators (FNO), including:

- 1D-FNO waveform-based detection
- 2D-FNO spectrogram-based detection

This repository is organized for reproducibility of the paper experiments while excluding restricted datasets and large model artifacts.

## Repository Layout

- `src/lunar_fno/models/fno1d.py`: 1D-FNO model used for waveform detection.
- `src/lunar_fno/models/fno2d.py`: 2D-FNO model used for spectrogram detection.
- `src/lunar_fno/train/train_waveform.py`: multi-seed waveform training/evaluation entrypoint.
- `src/lunar_fno/train/train_spectrogram.py`: multi-seed spectrogram training/evaluation entrypoint.
- `configs/`: default YAML configs.
- `docs/data_contract.md`: required NPZ keys and array shapes.
- `docs/reproducibility.md`: run commands and expected metrics.
- `expected_results/`: frozen seed-wise reference metrics from source notebooks.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run

Waveform (1D-FNO):

```bash
python -m lunar_fno.train.train_waveform --config configs/waveform_default.yaml
```

Spectrogram (2D-FNO):

```bash
python -m lunar_fno.train.train_spectrogram --config configs/spectrogram_default.yaml
```

## Outputs

Each training script writes under `output_dir` from its config:

- `models/best_model_seed{seed}.pth`
- `models/final_model_seed{seed}.pth`
- `metrics_eq.csv`
- `metrics_mq.csv`
- `training_history_seed{seed}.csv`

## Data Policy

This repository does not include datasets (`.npz`, `.mseed`) or checkpoint files.
See `docs/data_contract.md` for required keys/shapes and path conventions.

## Source Notebook Mapping

Waveform source pipeline:
- `/home/g202210640/wrk/MQ/new_work/wav_results/FNO_MQ_waveform_v02.ipynb` (cell 33)

Spectrogram source pipeline:
- `/home/g202210640/wrk/MQ/new_work/spec_results/FNO_MQ_spectrogram_v02.ipynb` (cell 31)

## Citation

See `CITATION.cff`.

## License

MIT License (`LICENSE`).
