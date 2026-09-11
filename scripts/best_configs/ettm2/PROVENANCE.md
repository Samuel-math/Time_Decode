# ETTm2 historical launch recovery

The original launch is recorded in local session `rollout-2026-09-01T03-01-12-01a05932-6e1c-7ac1-95f0-06be52e29090.jsonl`. Earlier searches missed its JSON-encoded terminal output.

At 2026-08-31 21:14:58 UTC, the running codebook process explicitly identifies run `ettm2_sota_tuned_20260901_051444` and parameters: context 512, batch 64, epochs 50, LR 0.0003, weight decay 0.0001, EMA decay 0.95, patch 8, embedding 64, compression 4, codebook 256, hidden/residual hidden 128, two residual layers and two quantizers, MLP, lowpass enabled. Sparse weight 0.3, noise amplitude 0.05, order/orthogonal weights 0.01, orthogonal start 0 and warmup 2. Channel order 2,5,6,4,0,3,1; group 0.

The script was read at 21:13:05 and 21:13:16 UTC, copied, patched only for run-prefix/retention/logging/GPU settings, uploaded and launched at 21:14:44 UTC. The 22:14:02 UTC supervisor-log read confirms this run completed codebook training (stopped epoch 39/50; best validation approximately 0.1077), then pretraining (best validation approximately 0.9467). The saved pretraining filename identifies context 336, step 6, lowpass and timefilter top-k 4. The original script specifies batch 64, LR 0.0003, 100 epochs, 3 layers, 4 heads, FF 128 and dropout 0.15.

The same supervisor log links the second profile to `ettm2_sota_tuned_20260901_055045`: codebook retrained; lowpass disabled; predictive context 672, causal transformer, FF 336, dropout 0.1. The original script specifies pretraining batch 64, LR 0.0003, 100 epochs, step 6 and prediction chunk 6. Thus these settings are linked to an actual launch, not merely a later edited wrapper.

Subsequent expert pretraining for the retained 96/336 fine-tuning results is separately recorded in `run_ettm2_expert_pretrain_fixed96_r15.sh` and the running command at 2026-09-04 19:35:10 UTC: experts 4 x 32, batch 128, context 336, and the first run's codebook. Retained 192 uses the first no-expert pretrained model; retained 720 uses the second no-expert pretrained model.

This closes the previously reported primary codebook and base-pretraining provenance gaps. The complete scratch entry is now assembled in `scripts/ettm2_best.sh` and `from_scratch.sh`. Each horizon uses a unique output directory and only newly generated stage checkpoints. Historical references are 96: 0.168354/0.245924; 192: 0.239478/0.294299; 336: 0.296795/0.327299; 720: 0.403429/0.398645 (MSE/MAE). The 96/336 branches are recovery_r16/e4d32; 192 is full_r1/h192_s8; 720 is 720_r5/c96d12t7.

Shell syntax, Python parser flag names and all four three-stage dependency chains were checked with an isolated test double. All 12 stage calls fixed seed 42; all fine-tuning contexts were 96; an injected historical PRETRAINED_MODEL path was ignored by the scratch entry. This checks script wiring only, not numerical reproduction. Actual retraining and metric comparison remain pending.
