# lunar-seismic-fno

Code for two seismic event detection pipelines built with Fourier Neural Operators (FNO):

- 1D FNO on waveform windows
- 2D FNO on spectrogram patches

The repository is designed for reproducible training and evaluation, while keeping data and model artifacts out of version control.

## What is included

- `src/lunar_fno/models/`: 1D and 2D FNO classifier definitions.
- `src/lunar_fno/data/`: dataset adapters for waveform and spectrogram NPZ files.
- `src/lunar_fno/train/`: training entrypoints for each modality.
- `configs/`: YAML configs for training, model, runtime, and paths.
- `docs/data_contract.md`: required NPZ keys and accepted shapes.
- `docs/reproducibility.md`: end-to-end run steps and reproducibility notes.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Data handling

Keep the NPZ files on Figshare and download them locally before training.
Do not commit dataset files to Git.

Figshare record:

- <https://doi.org/10.6084/m9.figshare.30209080.v1>

Required files:

- `combined_EQ8_MQ64.npz` (waveform EQ data)
- `EQ_event_noise_data_globalnorm8.npz` (spectrogram EQ data)
- `MQ_TPTN_plusFPFN_k4.npz` (MQ data used by both pipelines)

Why this is the better option for your case:

- your dataset bundle is large (~0.6 GB),
- Git history gets bloated if large binaries are versioned directly,
- Figshare gives stable hosting, DOI support, and easier citation for paper artifacts.

After download, point config paths to your local files:

- `configs/waveform_default.yaml`
- `configs/spectrogram_default.yaml`

## Run training

Waveform model:

```bash
python -m lunar_fno.train.train_waveform --config configs/waveform_default.yaml
```

Spectrogram model:

```bash
python -m lunar_fno.train.train_spectrogram --config configs/spectrogram_default.yaml
```

You can also use the helper scripts:

```bash
bash scripts/run_waveform.sh
bash scripts/run_spectrogram.sh
```

## Output files

Each run writes to `paths.output_dir` in the selected config.
For each seed, the scripts save:

- `models/best_model_seed{seed}.pth`
- `models/final_model_seed{seed}.pth`
- `training_history_seed{seed}.csv`

And per modality:

- `metrics_eq.csv`
- `metrics_mq.csv`
- `early_stopping.csv`
- ROC and confusion matrix figures under `EQ_results/` and `MQ_results/`

## Practical checklist

1. Create and activate `.venv`.
2. Install package with `pip install -e .`.
3. Download NPZ files from Figshare.
4. Update `paths.eq_npz` and `paths.mq_npz` in config YAML.
5. Run waveform and/or spectrogram training command.
6. Check `output_dir` for metrics CSVs and saved models.

## Citation

See `CITATION.cff`.

## License

MIT (`LICENSE`).
