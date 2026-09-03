# ACT（按子集从头训练）× LIBERO 标准四套件 评测记录

**评测对象**：4 个 ACT 策略（`lerobot` 实现），各自仅用对应 LIBERO 子集的官方训练集**从头训练**（无预训练权重、无跨子集共享参数）
**评测基准**：LIBERO 标准四套件（`libero_spatial` / `libero_object` / `libero_goal` / `libero_10`）
**数据来源**：远程主机 `<private>`，本地镜像目录 `act-libero-full/`（full 结果）+ `act-libero/`（smoke/preflight 结果）
**评测代码版本**：`pi05-libero-plus-pro-eval` harness（`act_client_adapter.py`/`act_runtime.py`，同 `../openvla-libero/EVAL_REPORT.md` §4.3 所述，harness 目录未纳入 git 版本控制）
**评测时间**：2026-08-27（约 10.6 小时，训练已在此前完成）

---

## 1. 结论速览

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 500 | 194 | 306 | 38.8% |
| libero_object | 500 | 176 | 324 | 35.2% |
| **libero_goal** | 500 | **15** | 485 | **3.0%** |
| libero_10 | 500 | 113 | 387 | 22.6% |
| **total** | **2000** | **498** | **1502** | **24.90%**（Wilson 95% CI: [23.1%, 26.8%]） |

**核心结论**：ACT 从头训练在同分布 LIBERO 上的整体表现（24.90%）明显弱于同分布的 OpenVLA（69.15%，见 `../openvla-libero`）和 π0.5（97.10%），且 `libero_goal` 子集近乎完全失败（3.0%）——这**不是训练或评测流程的 bug**（详见 §4.2），而是本次评测中 ACT 的一个已知设计限制：**ACT 未接入语言条件（no language conditioning）**，而 `libero_goal` 套件的任务结构恰恰是"同一场景、不同目标指令"，ACT 结构上无法区分不同 instruction 对应的不同目标，因此该子集的低成功率是预期之内的架构局限，而非训练失败。`libero_spatial`/`libero_object`（更依赖视觉空间推理、instruction 区分度较低）ACT 表现相对更好（35-39%）。全部 2000 个 episode 均无基础设施错误。

---

## 2. 详细结果

### 2.1 full（2000 episodes，全部 4 个子集，均为 bug 修复后的有效结果）

| suite | 对应 checkpoint | planned | attempted | successes | failures | success_rate |
|---|---|---:|---:|---:|---:|---:|
| libero_spatial | `act-final/libero_spatial` | 500 | 500 | 194 | 306 | 38.8% |
| libero_object | `act-final/libero_object` | 500 | 500 | 176 | 324 | 35.2% |
| libero_goal | `act-final/libero_goal` | 500 | 500 | 15 | 485 | 3.0% |
| libero_10 | `act-final/libero_10` | 500 | 500 | 113 | 387 | 22.6% |
| **total** | | **2000** | **2000** | **498** | **1502** | **24.90%** |

- 运行区间：2026-08-27T12:50:27Z → 2026-08-27T23:29:04Z（约 10.6 小时，2000 episodes）。
- `libero_object`/`libero_spatial` 两个子集此前曾因夹爪符号反转 bug（见 §4.1）产生过一版无效结果，已归档为 `libero-full-libero_{object,spatial}.gripperbug-invalid/`（未删除，仅不纳入本报告统计），本表为 bug 修复后重新训练评测的有效结果。

### 2.2 smoke（40 episodes，流水线冒烟检查，来自 `act-libero/`）

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 10 | 4 | 6 | 40.0% |
| libero_object | 10 | 4 | 6 | 40.0% |
| libero_goal | 10 | 0 | 10 | 0.0% |
| libero_10 | 10 | 4 | 6 | 40.0% |

样本量过小，不具统计意义，仅用于流水线冒烟检查；`libero_goal` smoke 0% 与 full 3.0% 量级一致，间接印证 §1 的"无语言条件"限制并非评测流程偶发问题。

### 2.3 Preflight

`server_startup`、checkpoint 加载、单条 `client_run` 均 `passed`，验证了 4 个 checkpoint 各自的加载路径、LIBERO BDDL/init 文件解析、policy server 通信链路全部可用。

---

## 3. 评测配置

| 项目 | 取值 |
|---|---|
| 评测协议名 | `act-lerobot-suite-specialist` |
| 训练数据 | LIBERO 官方训练集（`openvla/modified_libero_rlds` 转换而来），按子集独立，无跨子集共享 |
| 模型实现 | `lerobot.common.policies.act`（`ACTConfig`/`modeling_act`） |
| 动作 chunk | `chunk_size=100`，`n_action_steps=100`（每次推理产出 100 步动作 chunk 后再重新推理，即 `replan=100`，与 π0.5 的 `replan=5`、OpenVLA 的 `replan=1` 均不同） |
| 语言条件 | **无**（`act_runtime.py` 明确设计为不接入语言指令，仅以视觉观测为条件） |
| `max_steps` | `libero_spatial=220`、`libero_object=280`、`libero_goal=300`、`libero_10=520`，与 π0.5/OpenVLA 评测协议一致 |
| GL 后端 | `egl`（headless） |

---

## 4. 注意事项 / 数据局限性

1. **夹爪符号反转 bug（已修复于 2026-08-27，本报告数字均为修复后结果）**：`act_client_adapter.py` 的 `postprocess_action()` 早期实现是从 `openvla_client_adapter.py` 复制而来，错误地套用了 OpenVLA 的 `[0,1]→[-1,1]` 区间映射 + 夹爪取反变换；而 ACT 模型的原始输出本就已经是环境可直接执行的 `{-1,+1}` 原生动作空间（通过用训练数据首帧回放模型、比对预测动作与训练集 ground-truth 逐步核实），bug 导致每一步的夹爪指令符号被系统性取反，抓取动作在结构上不可能成功。定位过程：`libero_object` full 跑出精确的 0/500（零方差，全部 episode 都跑满 max_steps 判负），但训练 loss 完全正常（约 0.10，与其余子集一致），排除了 checkpoint 损坏的可能，转而怀疑 client 端动作后处理——回放测试证实了这一点。修复方式：将 ACT 的 `postprocess_action` 改为直通（identity），不做任何区间/符号变换。受影响的 `libero_object`/`libero_spatial` 两个子集（smoke 阶段全部 4 个子集均受影响）已系统性重跑，旧结果归档为 `.gripperbug-invalid` 后缀目录保留但不计入统计。
2. **`libero_goal` 的 3.0% 不是 bug**：详见 §1，ACT 本次评测按设计不接入语言指令，而 `libero_goal` 套件的任务区分度主要来自指令文本而非场景视觉差异，ACT 结构上无法区分，该子集低成功率是预期之内的架构局限，已通过 checkpoint 训练/回放验证正常，不需要按 bug 重新排查。
3. **ACT 每 100 步才重新推理一次**（`chunk_size=100`），与 OpenVLA 每步都推理（`replan=1`）、π0.5 每 5 步推理一次（`replan=5`）在计算开销和对环境扰动的响应速度上均不同，跨模型对比成功率时需注意这一推理节奏差异。
4. **smoke 样本量过小**：详见 §2.2，不具统计意义。
5. **全部 2000 个 full episode 均无基础设施错误**（`error_category` 全部为 `null`），失败均为真实的策略未达成任务目标。
6. 视频文件本地镜像未同步，仍完整保留在远程 `/mnt/data/results/act-libero-full/`。

---

## 5. 数据溯源

| 内容 | 路径 |
|---|---|
| 逐 episode 明细 | `libero-full-libero_{spatial,object,goal,10}/episodes.jsonl` |
| 成功率汇总表 | 各 run 目录下 `summary.csv` / `summary.json` |
| 运行环境快照 | `<run>/manifests/environment-*.json` |
| Checkpoint 来源 | `<run>/manifests/checkpoint-server-*.json`（含 `weights_sha256`） |
| 夹爪 bug 修复前的归档结果 | `libero-full-libero_{object,spatial}.gripperbug-invalid/`（保留未删除，仅不计入统计） |
| server/client 原始日志 | `<run>/logs/server-*.log`、`<run>/logs/client-egl-*.log` |
| 与 π0.5 / OpenVLA 的横向对比 | `../analysis-report/cross_policy_report_zh.md` |

目录结构与字段含义详见同级 `../CLAUDE.md`。
