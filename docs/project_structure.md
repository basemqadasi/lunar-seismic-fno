# Project Structure

- `src/lunar_fno/models`: model definitions for 1D and 2D FNO classifiers.
- `src/lunar_fno/data`: dataset adapters for waveform and spectrogram NPZ arrays.
- `src/lunar_fno/train`: CLI entrypoints for multi-seed training/evaluation.
- `src/lunar_fno/utils`: reproducibility, IO contracts, and metrics helpers.
- `configs`: YAML configs for both modalities.
- `expected_results`: fixed metrics from source notebooks.
