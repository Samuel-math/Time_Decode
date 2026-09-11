# Time DeCode

## Environment

Run from the repository root on a machine with an NVIDIA GPU and a compatible driver.

```bash
conda create -n time_decode python=3.10.8 -y
conda activate time_decode
pip install -r requirements.txt
```

## Forecasting Quick Start

Place the dataset CSV files in `datasets/`: `ETTm1.csv`, `ETTm2.csv`, `ETTh1.csv`, `ETTh2.csv`, `weather.csv`, `electricity.csv`, and `traffic.csv`.

Run codebook training, pre-training, and forecasting fine-tuning from scratch:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/single_run.sh \
  --dset etth2 --input_len 96 --output_len 336 --force --no_resume
```

Dataset options: `ettm1`, `ettm2`, `etth1`, `etth2`, `weather`, `electricity` (`ecl`), and `traffic`. Forecast horizons: `96`, `192`, `336`, and `720`.

Logs and metrics are saved in `logs/`; checkpoints are saved under `vqvae-only/saved_models/` and `decoder_only_NTP/saved_models/`.

Per-dataset retained configurations are available through `scripts/*_best.sh`; see [configuration status](scripts/best_configs/README.md). Historical-best reproduction is still being verified, and some entries require existing checkpoints.
