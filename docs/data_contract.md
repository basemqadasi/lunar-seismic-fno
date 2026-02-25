# Data Contract

This project reads NPZ files for waveform and spectrogram pipelines.
Use the keys and shapes below.

## Waveform pipeline (1D FNO)

Required keys:

- `waveform_data`
- `waveform_labels`

Accepted `waveform_data` shapes:

- `(N, L)`
- `(N, L, 1)`
- `(N, L, 1, 1)`

Internal handling:

- data is canonicalized and fed as waveform sequences,
- labels are binary with shape `(N,)` and values in `{0, 1}`.

## Spectrogram pipeline (2D FNO)

Required keys:

- `spectrogram_data`
- `spectrogram_labels`

Accepted `spectrogram_data` shapes:

- `(N, H, W)`
- `(N, H, W, 1)`

Internal handling:

- spectrogram arrays are canonicalized to `(N, H, W, 1)`,
- labels are binary with shape `(N,)` and values in `{0, 1}`.

## Config path fields

Set file locations in YAML configs under:

- `paths.eq_npz`
- `paths.mq_npz`

And make sure key names match:

- `paths.x_key`
- `paths.y_key`

## Recommended data workflow

Keep datasets in Figshare (or another data repository) and download locally.
Do not track NPZ files in Git history.

Figshare record used for this project:

- <https://doi.org/10.6084/m9.figshare.30209080.v1>

Required NPZ filenames:

- `Combined_EQ_MQ64_data.npz`
- `EQ_event_noise_data_.npz`
- `PSE_MQ_test_data.npz`
