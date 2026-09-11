# Recovered ECL training chains

The original host `a5d042a72c-c63ecbef` (port 15096) was reopened CPU-only.
`recovered_stage_metadata.json` contains the actual saved args/config of both
original codebook and pretraining checkpoints and the historical best ECL-192
fine-tuning checkpoint. These are not inferred from wrapper defaults.

| Retained horizon | Upstream run | CB context | CB LR | PRE context / LR |
| --- | --- | --- | --- | --- |
| 192 | 20260829_021235 | 512 | .001 | 128 / .001 |
| 96, 336, 720 | 20260831_020025 | 128 | .0003 | 128 / .001 |

Both CBs use batch 64, maximum 50 epochs, seed 42, EMA .95, sparse .3/.05,
order .01, orth .01 with start 0/warmup 2. They differ only in context, LR and
output path in their saved args. PRE uses batch 32, maximum 100 epochs,
step/pred 6/6, SN 20/.5/1, layer loss weights 1/.5, frozen tokenizer, no experts.
PRE args differ only in input/output checkpoint paths.

For each upstream pair, all 12 encoder, 12 decoder and 6 VQ tensors match
exactly between the codebook checkpoint and the pretrained model. This verifies
the tokenizer dependency. Historical CB best losses are .01705770374142698 and
.016327741821961745; PRE best losses are 1.2434907977173968 and 1.297015624802287.

ECL-192 FT: context 96, batch 32, LR .001, weight decay .0001, Huber 2,
tau .8, step/pred 8/10, frozen decoder, no experts, seed 42, maximum 50 epochs.
Its checkpoint epoch is zero-based 12. Historical test is
.16089020669460297 / .2510242462158203. The other retained FT recipes remain
unchanged, except that scratch runs route their output into fresh directories.

`bash scripts/ECL_best.sh` now trains every horizon from a fresh codebook through
pretraining, fine-tuning and final test. No historical checkpoint is consumed.
All FT contexts remain 96. Original CB context 512 for horizon 192 is preserved;
it must not be confused with the FT context. Each run has its own directory,
logs, environment record and source commit. Historical weights are never replaced.

The original CB/PRE, August 29 FT checkpoints and logs were copied to a local
46 MB backup, `time_decode_server_backups/ecl_original_stages_20260912.tar.gz`
(outside Git); source/destination SHA-256:
`64b7e324c288bf427de1721a45de6fee984e2435a2ca25cfa8c65010833b6b6e`.

Parameter recovery and shell wiring are not numerical reproduction. GPU reruns
must still compare final test results before any reproduction pass is claimed.
