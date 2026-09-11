# ETTh2 recovered scratch recipe

Status: candidate; numerical reproduction remains unverified.

Codebook: full startup arguments captured in this thread at 2026-09-02 01:56:19 UTC for `etth2_sota_tuned_20260902_010250`. The `etth2_freq_m7_base1_20260831_011724` codebook has identical arguments except save_path. Context 128, patch 8, two residual codebooks, sparse 0.3 / amplitude 0.05, order/orthogonal weights 0.01, warmup 5.

Predictive pretraining: archived `scripts/etth2_sota_tuned.sh` and `run_etth2_expert_pretrain_grid.sh`: context 296, batch 64, progressive step 3, prediction chunk 6, 3 layers / 4 heads / FF 256, causal transformer, LR 0.0003. Horizons 192/336 use experts 3 x 32. For the no-expert branch, remaining parser defaults were recovered from the contemporary pretraining startup argument dump and the archived pipeline; this is not a direct dump of the selected pretrained checkpoint. Channel order is fixed to 6,5,4,2,1,0,3. Effective codebook EMA decay is 0.95.

Horizon 96 first repeats `search_etth2_cb128_96_r2/h` (full startup args captured 2026-09-01 18:08:53 UTC), then trains the internal residual-only branch `k31224g3`. This is a sequential parameter-training chain, not prediction fusion. The residual branch receives only the freshly trained base fine-tuning checkpoint.

Other fine-tuning branches are the retained h192, d5lr75s8p16 (336), and d5lr5p6 (720) scripts. All stages use seed 42; fine-tuning context stays 96. Unique scratch directories retain all stage checkpoints and logs.
