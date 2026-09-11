# ECL upstream trace

Status: incomplete historical invocation provenance, not a reproduction pass.

## Exact result anchor

The original September 2 result scan, confirmed again in the September 3 output,
records `0.16089020669460297 / 0.2510242462158203` in:

```
decoder_only_NTP/saved_models/patch_vqvae_finetune/electricity_freq_m321_base1_20260829_021235/electricity/patch_vqvae_finetune_cw96_tw192_model1_timefilterlitek128_grp0_results.csv
```

This proves the run name and fine-tuning context 96, not all training arguments.

## Recovered upstream progress for the other retained pretraining run

Original tool output at UTC 2026-09-06 04:02:44.048, from
`logs/electricity_freq_m321_base1_20260831_020025/_progress_snk20a0p5t1p0.log`:

```
[04:00:42] CB DONE g0_grp Epoch 13/50 | Train Loss: 0.0191 (Recon: 0.0040, VQ: 0.0075, Perplexity: 1.000) | Valid Loss: 0.0181 (Recon: 0.0054, VQ: 0.0094, Perplexity: 0.998)|>>> 早停: val_loss 连续 10 个 epoch 未下降|码本预训练完成！|最佳验证损失: 0.0163|
[11:58:47] PRE DONE g0_grp -> Best model saved (val_signal: 1.2970)|>>> 早停: val_loss 连续 5 epoch 未显著下降（min_delta=0.0001, smooth_k=1）|预训练完成。最佳验证损失: 1.2970|
```

The PRE model is:
`patch_vqvae_ps4_cb256_cd128_l3_in128_step6_model1_rvq2_timefilterlitek128_snk20a0p5t1p0_grp0.pth`.
These progress values belong to August 31, **not** the August 29 ECL-192 chain.
The later fine-tuning experiments reused these run directories. A later invocation
with CB/PRE SKIP does not prove the parameters of the earlier CB/PRE training.

## Traced source code

Git object `a2e81d7998aecbb967f446c4d48e091e8ce94a5a` contains the full chain:

1. `scripts/ECL_best.sh`, profile 192: CB context 512, batch 64, epochs 50,
   LR 3e-4, sparse weight .3/amplitude .05, order .01, orth .01/warmup 2;
   PRE context 128, batch 32, epochs 100, LR 1e-3, step/pred 6/6;
   FT context 96, batch 32, epochs 50, LR 1e-3, Huber 2, tau .8, step/pred 8/10.
2. `scripts/decoder_only_NTP/channel_group_pipeline.sh` dispatches codebook,
   pretraining and fine-tuning. The codebook command fixes EMA decay .95.
3. `vqvae-only/codebook_pretrain.py` defaults to seed 42; early-stop patience
   is 10; when orth is enabled, checkpoint selection starts at zero-based
   `orth_start + orth_warmup`.

This supplies a complete **source-defined candidate**, but no located launch
record binds that Git object/profile to the retained August 29 result. Archived
later ECL scripts instead specify CB context 128. Do not merge these versions or
call either the exact historical invocation without additional evidence.

## Exact missing artifacts to recover

From `electricity_freq_m321_base1_20260829_021235`:
`cb_g0_grp.log`, `pre_g0_grp_snk20a0p5t1p0.log`, and
`ft_g0_grp_tp192_snk20a0p5t1p0.log` (or unsuffixed equivalents),
or the corresponding stage checkpoints containing saved args/config.
For the retained August 31 upstream chain, recover its CB and PRE args as well.

The two rebooted hosts, local code archives, Git history, rollout tool outputs
and the local read-only thread-history database have been searched. The former
ports 15096, 18666, 31381 and 24952 all refused TCP connections during this trace.
Their old data disks/complete backups remain the direct recovery route.
