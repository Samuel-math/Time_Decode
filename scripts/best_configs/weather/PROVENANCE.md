# Weather from-scratch recovery

Status: recovered recipe, not yet verified against historical test metrics.

- Codebook arguments were read from `weather_sota_search_r1_20260903_210451/weather/codebook_ps8_cb512_cd128_rvq2_dlp_model1_grp0.pth` on server B. Codebook context is 672, orthogonal warmup is 2; checkpoint saving starts at epoch 2.
- Pretraining arguments for horizons 192/720 were read from `search_weather_expert_pretrain_ctx672_r5/e3d48`; horizon 336 uses `search_weather_expert_pretrain_ctx672_r5/e4d32`.
- Horizon 96 uses the archived `run_weather_expert_pretrain.sh` with context 96, batch 64 and experts 4 x 32, corroborated by the original running command recorded in this thread at 2026-09-04 21:50:04 UTC. Its original pretrained checkpoint is not currently available for a direct argument comparison.
- Fine-tuning: archived `run_weather_expert_fixed96_r6.sh` for 96/192/720 and `run_weather_sparse_experts_fixed96_r21.sh` branch `w336_top2` for 336. Fine-tuning context is always 96; all stages use seed 42.

Historical reference pairs (MSE/MAE): 96: 0.155971/0.198296; 192: 0.204557/0.242019; 336: 0.261269/0.286156; 720: 0.338068/0.333706.

`scripts/weather_best.sh` now creates a unique run directory for each horizon and trains every stage from scratch. It does not import any historical checkpoint. The per-horizon fine-tuning helpers receive only the checkpoint created in that run. Complete stage weights, logs, environment and source diff remain under `scratch_runs/weather/` even when metrics differ.
