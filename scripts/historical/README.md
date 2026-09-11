# 历史脚本原件与日志证据

这里是溯源归档，不是当前最优复现入口。当前入口仍是仓库 `scripts/*_best.sh`。

`retained_20260908/` 收录原 A 机 9 月 8 日备份，以及此前整理的 B/C 机脚本快照。目录名沿用 A 机备份日期，不表示 B/C 快照也是当天生成。49 份 Shell 文件均与来源文件逐字节一致，SHA256 见 `manifest.json`，均通过 `bash -n`。这只证明归档完整和语法正确，不证明从头训练复现了历史数值。

| 数据集 | A 机历史主脚本 |
|---|---|
| ETTm1 | [ettm1_best.sh](retained_20260908/original_A/scripts/ettm1_best.sh) |
| ETTm2 | [ettm2_best.sh](retained_20260908/original_A/scripts/ettm2_best.sh) |
| ETTh1 | [etth1_best.sh](retained_20260908/original_A/scripts/etth1_best.sh) |
| ETTh2 | [etth2_best.sh](retained_20260908/original_A/scripts/etth2_best.sh) |
| Weather | [weather_best.sh](retained_20260908/original_A/scripts/weather_best.sh) |
| ECL | [ECL_best.sh](retained_20260908/original_A/scripts/ECL_best.sh) |
| Traffic | [traffic_best.sh](retained_20260908/original_A/scripts/traffic_best.sh) |

另保留 B/C 机各七个历史主脚本、A 机 ECL/ETTm2/Traffic/Weather 的16个分长度脚本，以及留存的 ECL 后续微调、Traffic 无专家微调、ETTm2 专家预训练和两份 sota_tuned 脚本。

## ECL 日志恢复范围

[electricity_log_excerpts.json](retained_20260908/evidence/electricity_log_excerpts.json) 保存四次聊天工具输出中的101行日志/结果原文，以及来源会话和 UTC 时间。这是恢复的片段，不是完整原始日志文件：

- 2026-09-03 21:52:28：逐个读取历史 results.csv，明确记录 `20260829_021235` 的192结果为 MSE 0.16089020669460297、MAE 0.2510242462158203。
- 2026-09-05 02:03:10：720 的启动参数和加载配置。
- 2026-09-05 10:47:31：336 的启动参数及运行过程。
- 2026-09-06 11:58:19：720 的 pipeline 过程、启动参数及历史预训练验证损失 1.2434907977173968。

不能拿336或720的启动参数替代192。原始码本/预训练完整启动日志，以及192完整微调 args，仍未在本次恢复材料中找到。

## ECL 脚本版本差异

| 配置 | A机 apart/ECL_192.sh | A机合并 ECL_best.sh 的192分支 |
|---|---:|---:|
| CB_CONTEXT_POINTS | 512 | 128 |
| CB_BATCH_SIZE / CB_LR | 64 / 3e-4 | 64 / 3e-4 |
| PRETRAIN_CONTEXT_POINTS | 128 | 128 |
| FINETUNE_CONTEXT_POINTS | 96 | 96 |
| FINETUNE_LR / HUBER_DELTA | 1e-3 / 2.0 | 1e-3 / 2.0 |
| FORECAST_STEP_SIZE / PRED_LEN | 8 / 10 | 8 / 10 |

这些是留存脚本的明确值，但尚不能据此把任一版本确认为192历史最优的完整训练配置。原件保留双方版本，不通过修改历史文件消除差异。

## 使用限制

不要直接在归档目录执行这些脚本：原件可能包含旧 GPU 编号、服务器绝对路径、历史 checkpoint 依赖、相对根目录假设和保留/清理逻辑。它们需要原运行目录和对应 pipeline，不是独立可运行的软件包。复现脚本应从证据中另行构建，固定 seed=42、微调上下文96，并通过真实重训核验；归档不覆盖当前脚本、权重或运行日志。
