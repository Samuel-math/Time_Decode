# Retained forecasting configurations

Configurations are grouped by dataset and forecast horizon. All retained forecasting contexts are 96 and seeds are 42.

| Dataset | Entry | Current status |
|---|---|---|
| ETTm1 | `scripts/ettm1_best.sh` | Full-pipeline candidates; fresh-environment reproduction queued |
| ETTm2 | `scripts/ettm2_best.sh` | Matched codebook/pretraining launch records and retained best fine-tuning branches; complete scratch entry assembled, numeric reproduction pending |
| ETTh1 | `scripts/etth1_best.sh` | Full-pipeline candidates; 96 reproduction queued |
| ETTh2 | `scripts/etth2_best.sh` | Recovered full-pipeline candidate; 96 includes fresh base fine-tuning before residual training; numerical reproduction pending |
| Weather | `scripts/weather_best.sh` | Recovered full-pipeline candidate; fresh rerun pending, see weather/PROVENANCE.md |
| Electricity | `scripts/ECL_best.sh` | All four complete scratch chains recovered from original CB/PRE/FT evidence; numerical reproduction pending |
| Traffic | `scripts/traffic_best.sh` | Historical no-expert checkpoint-based fine-tuning parameters; original pretraining provenance pending |

Use `HORIZONS="96 192"` to select horizons. For checkpoint-dependent entries, explicitly set `TD_STAGE=finetune`; the default refuses to claim or run an incomplete from-scratch recipe. Keep the required checkpoint at its recorded relative path, or use `PRETRAINED_MODEL` where supported. This is not checkpoint download automation.

Example full-pipeline candidate:

```bash
HORIZONS=96 bash scripts/etth1_best.sh
```

Example historical checkpoint-based fine-tuning:

```bash
TD_STAGE=finetune HORIZONS=720 bash scripts/traffic_best.sh
```

No configuration in this commit is certified to reproduce the historical test metrics from scratch. Evaluation will record actual differences; passing requires both reported metrics to match to six decimal places. Original weights and failed attempts must be retained separately. These scripts may train for many epochs; syntax validation alone is not a training or reproducibility test.
