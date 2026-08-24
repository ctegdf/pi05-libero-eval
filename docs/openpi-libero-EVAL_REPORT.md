# π0.5 (pi05) × LIBERO 基础 Benchmark 评测记录

**评测对象**：`openpi` 仓库下的 π0.5 基准权重（base）与 LIBERO 微调权重（fine-tuned）
**评测基准**：LIBERO 标准四套件（`libero_spatial` / `libero_object` / `libero_goal` / `libero_10`）
**数据来源**：远程主机 `<private>`，本地镜像目录 `openpi-libero/`
**评测代码版本**：`openpi` git commit `15a9616a`（**工作区含 13 个未提交改动**，见「注意事项」）
**评测时间**：2026-08-10

---

## 1. 结论速览

| 权重/协议 | Smoke 结果 | Full 结果 | 结论 |
|---|---|---|---|
| **official**（`pi05_libero` 微调权重） | 7/8 = 87.5% | **1942/2000 = 97.1%** | 4 套件全部高成功率，达到预期性能 |
| **base-libero-assets**（`pi05_base` 基准权重 + `pi05_libero` 归一化统计） | 0/8 = 0% | **0/2000 = 0.0%** | 每个 episode 均在 `max_steps` 耗尽后判定失败，无一成功（见 §4） |
| **base-native**（`pi05_base` 基准权重 + 自带统计） | — | **未运行，流水线在 checkpoint 校验阶段报错** | `pi05_base` 缺少 LIBERO 专用 `norm_stats.json`，协议不适用（见 §4） |

**核心结论**：LIBERO 微调对 π0.5 在该基准上的表现是决定性的——微调权重达到 97.1% 成功率，而未经 LIBERO 微调的基准权重（即便强行套用 LIBERO 的归一化统计）在全部 2000 个 episode 中零成功；若不提供 LIBERO 归一化统计则权重根本无法加载评测流程。

---

## 2. 详细结果

### 2.1 official（微调权重）— full

| suite | planned | attempted | successes | failures | success_rate |
|---|---:|---:|---:|---:|---:|
| libero_spatial | 500 | 500 | 493 | 7 | 98.6% |
| libero_object | 500 | 500 | 492 | 8 | 98.4% |
| libero_goal | 500 | 500 | 491 | 9 | 98.2% |
| libero_10 | 500 | 500 | 466 | 34 | 93.2% |
| **total** | **2000** | **2000** | **1942** | **58** | **97.1%** |

- `libero_10`（长程组合任务）失败率明显高于其余三套件（34/58 = 全部失败中的 59%），符合其任务难度更高的预期。
- 全部 58 例失败均无 `error_category`（非环境/系统错误），即策略在环境中完整运行完毕但未达成任务目标，属于真实的策略失败。
- 运行区间：2026-08-10 04:41:02Z → 10:13:30Z（约 5.5 小时，2000 episodes，含渲染视频写入）。

### 2.2 official（微调权重）— smoke

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 2 | 2 | 0 | 100% |
| libero_object | 2 | 2 | 0 | 100% |
| libero_goal | 2 | 1 | 1 | 50% |
| libero_10 | 2 | 2 | 0 | 100% |
| **total** | **8** | **7** | **1** | **87.5%** |

用途：正式跑 full 前的流水线冒烟检查（checkpoint 加载、server 启动、渲染写盘等），样本量小，不代表真实成功率。

### 2.3 base-libero-assets（基准权重 + LIBERO 归一化统计）— full / smoke

两次运行（full 2000 episodes、smoke 8 episodes）**成功率均为 0%**，且失败模式一致：

- 所有 episode 均无 `error`/`error_category`（非崩溃、非环境异常）。
- 所有 episode 的 `action_steps` 都精确等于该任务的 `max_steps`（220/280/300/520，与 official 组一致），即策略从未在时限内达成任务，触发超时判失败。
- 结论：这是**真实的策略行为结果**，而非评测流程故障——π0.5 基准权重未见过 LIBERO 微调数据，套用 LIBERO 的动作/状态归一化统计后仍完全无法完成任何 LIBERO 任务。可作为「不微调 baseline」的下界基线（0%），衬托 §2.1 的微调收益。

### 2.4 base-native（基准权重 + 权重自带统计）— smoke

**未产生任何 episode**，流水线在 `checkpoint_validation` 阶段即报错退出（`exit_code=2`）：

```
CheckpointError: base-native requires physical-intelligence/libero norm_stats at
.../pi05_base/assets/physical-intelligence/libero/norm_stats.json;
protocol is not applicable and this is a checkpoint/provenance error,
not a zero-success evaluation
```

即 `pi05_base` 官方发布包内并未随附 LIBERO 任务的归一化统计文件，`base-native` 协议因此在设计上不可行，**不能**与 §2.3 的 0% 混为一谈（后者是真实评测出的 0%，前者是配置/资产缺失导致的评测未执行）。

---

## 3. 评测配置

| 项目 | 取值 |
|---|---|
| 代码仓库 | `openpi`（π0.5 官方仓库） |
| Git HEAD | `15a9616a00943ada6c20a0f158e3adb39df2ccac`（= 各 run 要求的 `required_upstream_commit`） |
| 微调权重 checkpoint | `pi05_libero`（`config=pi05_libero`, `protocol=official`） |
| 基准权重 checkpoint | `pi05_base`（`config=pi05_libero`, `protocol=base-libero-assets`，即基准权重 + 微调权重的归一化资产拼装） |
| 渲染/图像分辨率 | `resize=224` |
| 重规划间隔 | `replan=5` |
| GL 后端 | `egl`（headless） |
| GPU | `CUDA_VISIBLE_DEVICES=0` 单卡 |

各 suite 的 `max_steps`：`libero_spatial=220`、`libero_object=280`、`libero_goal=300`、`libero_10=520`。四套件在 full/smoke/base-libero-assets 三组运行中的 `max_steps` 分布完全一致（220/280/300/520 各 500 例），确保跨组对比公平。

---

## 4. 注意事项 / 数据局限性

1. **工作区非纯净提交**：所有 run 的 `git-*.json` 均显示相同的 13 个未提交改动（`M scripts/train.py`、`M src/openpi/models/model.py`、`M src/openpi/training/config.py`、`D pyproject.toml`、`D uv.lock`，以及新增的 `examples/libero/pi05_libero_{client,server,support}.py`、`run_pi05_libero.py`、`test_pi05_libero_eval.py`、`start_server.sh`、`examples/franka_isaaclab/`、`src/openpi/policies/franka_isaaclab_policy.py`）。**评测实际运行的是「commit 15a9616a + 这些本地补丁」的代码，而非纯 upstream commit**——这些补丁正是本地新增的 LIBERO 评测脚手架（client/server/support），预期之内，但如需复现须一并应用这些改动。
2. **base-libero-assets 的 0% 不是 bug**：详见 §2.3，是基准权重的真实策略输出，全部 episode 干净跑满 max_steps 无一崩溃。
3. **base-native 未纳入对比**：因资产缺失在校验阶段即失败，样本量为 0，不能作为"基准权重下界"的证据（该角色已由 base-libero-assets 承担）。
4. **smoke 结果仅供流水线冒烟验证**，样本量（8 episodes）过小，不具统计意义，正式结论以 full 结果为准。
5. 全部 4000 个 full episode（official 2000 + base-libero-assets 2000）均有对应 `.mp4` 渲染视频落盘，路径见各自 `videos/` 目录及 `episodes.jsonl` 中的 `video` 字段，可用于人工复核个别失败案例。

---

## 5. 数据溯源

| 内容 | 路径 |
|---|---|
| 逐 episode 明细 | `official-full/episodes.jsonl`、`base-libero-assets-full/episodes.jsonl` 等 |
| 成功率汇总表 | 各 run 目录下 `summary.csv` / `summary.json` |
| 运行环境快照 | `<run>/manifests/environment-*.json` |
| 代码版本/工作区状态 | `<run>/manifests/git-*.json` |
| Checkpoint 来源 | `<run>/manifests/checkpoint-request-*.json` / `checkpoint-server-*.json` |
| 流水线阶段时间线 | `<run>/manifests/result-*.json`（`events` 数组） |
| server/client 原始日志 | `<run>/logs/server-*.log`、`<run>/logs/client-egl-*.log` |
| 渲染视频 | `<run>/videos/*.mp4` |

目录结构与字段含义详见同级 `../CLAUDE.md`。
