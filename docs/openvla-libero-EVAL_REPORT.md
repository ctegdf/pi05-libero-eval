# OpenVLA-7B（按子集微调）× LIBERO 标准四套件 评测记录

**评测对象**：`openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`（4 个官方发布的按 LIBERO 子集独立微调权重，**不是**单一统一权重，与 π0.5 的 `pi05_libero` 单权重协议不同）
**评测基准**：LIBERO 标准四套件（`libero_spatial` / `libero_object` / `libero_goal` / `libero_10`）
**数据来源**：远程主机 `<private>`，本地镜像目录 `openvla-libero/`
**评测代码版本**：`pi05-libero-plus-pro-eval` harness（本次新增的 `run_eval.py`/`eval_client.py`/`eval_support.py`/`openvla_client_adapter.py`/`openvla_runtime.py` 通用评测脚手架，见 §4 关于该目录 git 状态的说明）
**评测时间**：2026-08-26 ~ 2026-08-27（约 1.3 天）

---

## 1. 结论速览

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 500 | 383 | 117 | 76.6% |
| libero_object | 500 | 435 | 65 | 87.0% |
| libero_goal | 500 | 361 | 139 | 72.2% |
| libero_10 | 500 | 204 | 296 | 40.8% |
| **total** | **2000** | **1383** | **617** | **69.15%**（Wilson 95% CI: [67.1%, 71.1%]） |

**核心结论**：这是**同分布评测**——4 个 checkpoint 各自只在对应子集的官方 LIBERO 训练集上微调过，评测子集与训练子集一一对应，不存在分布外迁移问题。即便如此，OpenVLA 的整体成功率（69.15%）也明显低于同协议下 π0.5 `pi05_libero`（97.10%，见 `../openpi-libero/EVAL_REPORT.md`），且四个子集之间差异很大：`libero_object`（物体识别与抓取为主）表现最好（87.0%），`libero_10`（长程组合任务）明显最差（40.8%）——与 π0.5 的模式一致（π0.5 也是 `libero_10` 最低，93.2%），但 OpenVLA 的绝对水平和子集间落差都远大于 π0.5。全部 2000 个 episode 均无 `error_category`（0 例基础设施错误），失败均为真实的策略未达成任务目标。

---

## 2. 详细结果

### 2.1 full（2000 episodes，全部 4 个子集）

| suite | 对应 checkpoint | planned | attempted | successes | failures | success_rate |
|---|---|---:|---:|---:|---:|---:|
| libero_spatial | `openvla-7b-finetuned-libero-spatial` | 500 | 500 | 383 | 117 | 76.6% |
| libero_object | `openvla-7b-finetuned-libero-object` | 500 | 500 | 435 | 65 | 87.0% |
| libero_goal | `openvla-7b-finetuned-libero-goal` | 500 | 500 | 361 | 139 | 72.2% |
| libero_10 | `openvla-7b-finetuned-libero-10` | 500 | 500 | 204 | 296 | 40.8% |
| **total** | | **2000** | **2000** | **1383** | **617** | **69.15%** |

- 各 checkpoint 独立加载、独立 policy server 进程评测（`protocol=openvla-libero-suite-specialist`），子集间互不干扰。
- `libero_10`（10 个子任务顺序组合的长程任务）失败率最高，与其"单步重规划、无 action chunk"的推理方式（`replan=1`，每个环境步都重新调用一次 `/infer`）在长程任务上误差累积更明显的预期一致。
- 全部 617 例失败均无 `error_category`，即策略在环境中完整运行完毕但未达成任务目标。
- 运行区间：2026-08-26T13:25:01Z → 2026-08-27T19:43:47Z（约 1.3 天，2000 episodes，含渲染视频写入）。

### 2.2 smoke（40 episodes，流水线冒烟检查）

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 10 | 0 | 10 | 0.0% |
| libero_object | 10 | 9 | 1 | 90.0% |
| libero_goal | 10 | 8 | 2 | 80.0% |
| libero_10 | 10 | 5 | 5 | 50.0% |

用途：正式跑 full 前验证流水线（checkpoint 加载、server 启动、渲染写盘等）是否可用。样本量（每子集 10 例）过小，噪声很大——`libero_spatial` smoke 0% 与 full 76.6% 差异悬殊，**不能**像 `libero-x` 报告那样用 smoke 结果验证流水线代表性，仅作为纯粹的"能不能跑通"冒烟检查，正式结论一律以 full 为准。

### 2.3 Preflight

`server_startup`、checkpoint 加载、单条 `client_run` 均 `passed`，验证了 4 个 checkpoint 各自的加载路径、LIBERO BDDL/init 文件解析、policy server 通信链路全部可用，未产生正式 episode 记录。

---

## 3. 评测配置

| 项目 | 取值 |
|---|---|
| 评测协议名 | `openvla-libero-suite-specialist`（自定义命名，强调按子集独立加载 checkpoint，而非单一统一权重） |
| Checkpoint 来源 | HuggingFace `openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`，官方发布权重，本地下载后 `scp` 中转到远程 `/mnt/data/models/openvla/` |
| Vision backbone | `dinosiglip-vit-so-224px` |
| 语言模型 | `meta-llama/Llama-2-7b-hf` |
| 动作表示 | `n_action_bins=256`（离散化动作 token，OpenVLA 原生动作表示，与 π0.5 的连续 flow-matching 动作头不同） |
| 重规划间隔 | `replan=1`（**无 action chunk**，每个环境步都重新推理一次，与 π0.5 的 chunk 式动作预测（`replan=5`）在推理节奏上不同，对照 OpenVLA 官方 LIBERO 评测脚本核实一致） |
| `max_steps` | `libero_spatial=220`、`libero_object=280`、`libero_goal=300`、`libero_10=520`，与 π0.5 评测协议完全一致，跨模型可比 |
| GL 后端 | `egl`（headless） |
| Transformers 版本 | `4.40.1` |

---

## 4. 注意事项 / 数据局限性

1. **4 个 checkpoint 而非单一权重**：OpenVLA 官方并未发布覆盖全部 LIBERO 子集的统一权重，只有按子集微调的 4 个独立发布包（`openvla-7b-finetuned-libero-{spatial,object,goal,10}`）。本报告的"total 69.15%"是四个**不同模型**在各自子集上的加权汇总，不是一个模型的泛化能力——与 π0.5 单权重 97.10% 对比时需注意这一评测口径差异。
2. **`replan=1` 而非 chunk 式规划**：OpenVLA 原生按单步动作输出设计，每个环境步都重新调用一次策略推理，与 π0.5 的 chunk 式规划在计算开销和行为特性上不同，本报告数字反映的是"OpenVLA 官方发布权重按其原生协议评测"的结果，不是架构公平对比。
3. **harness 代码目录未纳入 git 版本控制**：`pi05-libero-plus-pro-eval` 当前是一个未初始化提交历史的工作目录（`git rev-parse HEAD` 报 `ambiguous argument 'HEAD'`），因此没有 commit hash 可供溯源，与 `openpi-libero`/`libero-x` 报告里"13 处未提交改动基于某个 upstream commit"的情况不同——**没有 upstream commit 可对照**，可复现性依赖于本次评测归档的文件快照本身（`manifests/environment-*.json` 记录了完整的 Python 包版本清单）。
4. **smoke 样本量过小、噪声大**：详见 §2.2，不具统计意义，不能用于验证流水线代表性，仅作冒烟检查。
5. **全部 2000 个 full episode 均有对应 `.mp4` 渲染视频落盘**，路径见 `videos/` 目录及 `episodes.jsonl` 中的 `video` 字段。

---

## 5. 数据溯源

| 内容 | 路径 |
|---|---|
| 逐 episode 明细 | `libero-full-libero_{spatial,object,goal,10}/episodes.jsonl` |
| 成功率汇总表 | 各 run 目录下 `summary.csv` / `summary.json` |
| 运行环境快照 | `<run>/manifests/environment-*.json` |
| Checkpoint 来源 | `<run>/manifests/checkpoint-request-*.json` / `checkpoint-server-*.json` |
| 流水线阶段时间线 | `<run>/manifests/result-*.json`（`events` 数组） |
| server/client 原始日志 | `<run>/logs/server-*.log`、`<run>/logs/client-egl-*.log` |
| 渲染视频 | `<run>/videos/*.mp4`（本地镜像未同步视频，仅同步 episodes/summary/manifests；视频仍在远程 `/mnt/data/results/openvla-libero/`） |
| 与 π0.5 的横向对比 | `../analysis-report/cross_policy_report_zh.md` |

目录结构与字段含义详见同级 `../CLAUDE.md`。
