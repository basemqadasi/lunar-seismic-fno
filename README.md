# lunar-seismic-fno
## Fourier Neural Operator for Moonquake Detection

Abstract: Moonquakes provide critical observations for probing the lunar interior, yet their analysis ishindered by the limited number of recordings and their inherently low signal‐to‐noise ratio (S/N). Conventionaldetection methods such as Short‐Term Average/Long‐Term Average (STA/LTA) perform poorly on lunar data,while standard deep learning models (e.g., CNN, LSTM) require large volumes of clean, fixed‐length inputs andoften generalize weakly across domains. To address these challenges, we investigate the Fourier NeuralOperator (FNO) for moonquake detection. To our knowledge, this study presents the first application of FourierNeural Operators to planetary seismic event detection, leveraging operator learning to remain data‐efficientunder limited labels and variable acquisition settings. Our training data set includes waveforms andspectrograms of events and noise windows. Cross‐domain generalizability is assessed under two regimes: (a)training on earthquakes only, and (b) training on earthquakes augmented with 64 labeled moonquake examples.Both models are evaluated on independent Apollo Passive Seismic Experiment (PSE) and Lunar SeismicProfiling Experiment (LSPE) records. Despite limited training data, the 1D and 2D models achieve highdetection performance, with F1‐scores of 0.96 and 0.99, respectively. Furthermore, the resolution‐invariantnature of FNO enables application to waveform or spectrogram inputs of arbitrary length and sampling rate.Compared to CNN‐based approaches, FNO models require fewer parameters, fewer training examples, lesscomputation time, and yield superior cross‐domain generalization while maintaining competitive accuracy.These results highlight FNO as a flexible, lightweight framework for real‐time lunar seismic monitoring, withclear potential for extension to other planetary data sets such as Mars InSight




This code for two Moonquake event detection pipelines built with Fourier Neural Operators (FNO):

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

NPZ files for training and validation data set can be downloaded from Figshare record:

- <https://doi.org/10.6084/m9.figshare.30209080.v1>

Required files:

- `Combined_EQ_MQ64_data.npz` (waveform EQ data)
- `EQ_event_noise_data_.npz` (spectrogram EQ data)
- `PSE_MQ_test_data.npz` (MQ data used by both pipelines)


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
