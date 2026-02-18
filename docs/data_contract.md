# Data Contract

This repository expects NPZ files with explicit keys for each modality.

## Waveform (1D-FNO)

Required keys:

- `waveform_data`
- `waveform_labels`

Accepted waveform_data shapes (before canonicalization):

- `(N, L)`
- `(N, L, 1)`
- `(N, L, 1, 1)`

Canonical internal shape:

- `(N, L)` in memory, then dataset returns tensors shaped `(L, 1)`.

Labels:

- binary labels in `{0,1}` with shape `(N,)`.

## Spectrogram (2D-FNO)

Required keys:

- `spectrogram_data`
- `spectrogram_labels`

Accepted spectrogram_data shapes:

- `(N, H, W)`
- `(N, H, W, 1)`

Canonical internal shape:

- `(N, H, W, 1)`.

Labels:

- binary labels in `{0,1}` with shape `(N,)`.

## Path Map (default configs)

Waveform config uses:

- EQ: `/home/g202210640/wrk/MQ/new_work/train_test_data/combined_EQ8_MQ64.npz`
- MQ: `/home/g202210640/wrk/MQ/new_work/train_test_data/MQ_TPTN_plusFPFN_k4.npz`

Spectrogram config uses:

- EQ: `/home/g202210640/wrk/MQ/new_work/train_test_data/EQ_event_noise_data_globalnorm8.npz`
- MQ: `/home/g202210640/wrk/MQ/new_work/train_test_data/MQ_TPTN_plusFPFN_k4.npz`
