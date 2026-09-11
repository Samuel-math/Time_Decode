# Historical Traffic checkpoint provenance

## Matched pretrained checkpoint

- Imported file: `imported_checkpoints/traffic_noexpert_20260831.pth`
- SHA256: `a02f1527d6dc5a749e98ce1c6dc7614a1ef9e6d5d3a99683e1a22a022fdc10c5`
- Original run: `traffic_freq_m862_base1_20260831_002453`
- Saved epoch field: `33` (stored index; not reinterpreted as a one-based epoch).
- Saved validation loss: `2.426885578119866`.
- Original codebook: `vqvae-only/saved_models/vqvae_only/traffic_freq_m862_base1_20260831_002453/traffic/codebook_ps8_cb256_cd128_rvq2_model1_grp0.pth`.

Source: checkpoint args read on 2026-09-11 from server port 31381; corroborated by original tool output in this thread at 2026-09-06 14:56:16 UTC and the pipeline log captured at 2026-09-06 15:18:42 UTC.

## Confirmed predictive pretraining parameters

Context 128; progressive step 6; prediction chunk 6; batch size 16; workers 0; seed 42; standard scaler; multivariate features; RevIN enabled. Maximum 100 epochs, learning rate 0.001, weight decay 0.0001. Early stopping patience 5, warmup 5, min_delta 0.0001, smoothing window 1.

Patch size 8; compression 4; embedding dimension 64; codebook size 256; residual VQ layers 2 with weights 1.0/0.5. Encoder/decoder hidden size 128, residual layers 2, residual hidden size 128; MLP VQVAE backbone; decoder lowpass disabled.

Temporal backbone `timefilter_lite`, top-k 8, temperature 1.0; 3 layers, 4 heads, feedforward dimension 512, dropout 0.1. Soft-neighbor k=20, alpha=0.3, tau=0.5. Tokenizer frozen, VQ weights loaded, EMA updates disabled during predictive pretraining; vq/reconstruction loss weights both zero. Checkpoint is the historical no-expert version, not the later expert pretraining experiment.

## Recovered codebook evidence

Original codebook config dumps for both `20260828_034528` and `20260831_002453` were captured at 2026-09-06 14:56:41 UTC. Both show EMA decay 0.95, epsilon 1e-5, commitment cost 0.25, random VQ initialization, sparse weight 0.3, sparse amplitude 0.05, and the same model dimensions above. Thus an EMA-decay difference is not supported by these records.

The original 20260831 pipeline log reports codebook training stopped at epoch 13/50 after 10 epochs without validation improvement; best validation loss was approximately 0.0645. It then ran predictive pretraining and reported best validation loss approximately 2.4269, matching the imported checkpoint.

The config dump does not itself establish all codebook optimizer, orthogonality/order-loss and checkpoint-selection settings. Historical wrapper/pipeline defaults are candidate evidence, not proof of the effective values for this run. These remaining fields and the original codebook weights must be checked before promoting the recipe to a verified from-scratch reproduction.
