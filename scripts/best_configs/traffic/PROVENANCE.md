# Traffic original training chain recovered

Original host: `bce24a9cd7-7011d79a`, port 24952.
Original run: `traffic_freq_m862_base1_20260831_002453`.
`recovered_stage_metadata.json` records actual checkpoint args, epochs, losses,
SHA-256 hashes and the CB-to-PRE tokenizer tensor comparison.

The original PRE SHA-256 is
`a02f1527d6dc5a749e98ce1c6dc7614a1ef9e6d5d3a99683e1a22a022fdc10c5`,
identical to the imported `traffic_noexpert_20260831.pth` used by retained FT
recipes. All 12 encoder, 12 decoder and 6 VQ tensors match the original CB.

CB: context 128, batch 64, LR .0003, maximum 50 epochs, EMA .95, random init,
patch 8, embedding 64, compression 4, codebook 256, two RQ layers, sparse .3/.05,
order .01, orth .01/start 0/warmup 2, seed 42. Saved zero-based epoch 2
(third epoch), validation loss .06454002418156181. Do not replace this with
the later context-336 codebook or force training to stop at epoch 3: reproduce
the original validation-based selection and patience instead.

PRE: context 128, batch 16, LR .001, maximum 100 epochs, step/pred 6/6,
layers 3, heads 4, FF 512, dropout .1, timefilter top-k 8, SN 20/.3/.5,
RQ layer weights 1/.5, frozen tokenizer and EMA updates disabled, no experts,
seed 42. Saved zero-based epoch 33, validation loss 2.426885578119866.

The original FT-96 saved on this host is NOT substituted for the later retained
best configurations. The existing four FT helpers keep their historical best
parameters and now receive the newly trained PRE via `PRETRAINED_MODEL`.
All use FT context 96 and seed 42. Outputs are isolated per scratch run.

`scripts/traffic_best.sh` now executes CB -> PRE -> FT -> test for each selected
horizon, without a historical checkpoint dependency. No core model code changed.
Shell wiring checks are not numerical reproduction; final GPU test comparisons
are still pending.

Original CB/PRE/FT checkpoint directories, logs and channel ordering were backed
up locally outside Git at
`time_decode_server_backups/traffic_original_stages_20260912.tar.gz`.
Source and destination archive SHA-256 match:
`1c7d43ba8775d17ed7301e5c2389205f16300eea3432cc074ed54808ebff2cc8`.
