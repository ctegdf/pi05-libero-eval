# OpenVLA-7B（按子集微调）× LIBERO-Plus / LIBERO-Pro Zero-Shot 评测记录

**评测对象**：`openvla/openvla-7b-finetuned-libero-{spatial,object,goal,10}`（与 `../openvla-libero/EVAL_REPORT.md` 同一批 4 个按子集微调权重）
**评测基准**：LIBERO-Plus（7 类视觉/环境扰动 × 5 档难度）+ LIBERO-Pro（4 类结构性扰动：语言语义替换 / 物体替换 / 位置交换 / 任务替换；`env` 扰动 cell 缺失，与 π0.5 评测口径一致，见 §3）
**数据来源**：远程主机 `<private>`，本地镜像目录 `openvla-plus-pro/`
**评测代码版本**：`pi05-libero-plus-pro-eval` harness（同 `../openvla-libero/EVAL_REPORT.md`，未纳入 git 版本控制，见该报告 §4.3）
**评测时间**：2026-08-28 ~ 2026-09-02（约 5.8 天，含 GPU 分片并行加速，见 §4）

---

## 1. 结论速览

**这是一次零样本（zero-shot）迁移评测，不是同分布评测**：4 个 checkpoint 只在标准 LIBERO 各自子集的训练数据上微调过，从未见过 LIBERO-Plus 的视觉/环境扰动或 LIBERO-Pro 的结构性扰动。

| Benchmark | Planned/Attempted | Successes | Failures | Success rate | 对照：π0.5 | 对照：OpenVLA 同分布（`../openvla-libero`） |
|---|---:|---:|---:|---:|---:|---:|
| LIBERO-Plus | 10,030 | 2,534 | 7,496 | **25.26%**（CI95 [24.4%, 26.1%]） | 83.65% | 69.15% |
| LIBERO-Pro | 8,000 / 10,000 | 2,707 | 5,293 | **33.84%**（CI95 [32.8%, 34.9%]） | 58.79% | 69.15% |

**核心结论**：OpenVLA 的零样本迁移能力比 π0.5 弱得多——LIBERO-Plus 上从同分布的 69.15% 跌到 25.26%（−43.9 个百分点），LIBERO-Pro 上跌到 33.84%（−35.3 个百分点）；以各自的同分布基线为参照，π0.5 的降幅是 −13.5pp（Plus）和 −38.3pp（Pro）——即 **π0.5 在 Plus 上比 OpenVLA 稳健得多（−13.5pp vs −43.9pp），但在 Pro 上两者降幅其实接近（−38.3pp vs −35.3pp）**，说明 LIBERO-Pro 的结构性扰动（尤其位置交换、任务替换）对两个策略都构成了相近量级的冲击，而 LIBERO-Plus 的视觉/环境扰动上 π0.5 展现出明显更强的鲁棒性优势。最突出的单点发现是 **LIBERO-Pro 的 `swap(position)`（物体位置交换）扰动导致 OpenVLA 在全部 2000 个 episode 上 100% 失败（0/2000）**——已核实这不是基础设施错误（无 `error_category`、全部 episode 都正常运行满 `max_steps` 后判超时失败、无崩溃），是一个真实的、灾难性的策略盲区：OpenVLA 的单图输入 + 逐步重规划（`replan=1`）架构似乎严重依赖训练时记忆的物体相对位置关系，一旦位置被交换就完全无法完成任务；相比之下 π0.5 在同一扰动上仍保有 28.10% 成功率，ACT 也有 0.35%（详见 `../analysis-report/cross_policy_report_zh.md`）。全部 18,030 个 episode 均无基础设施错误。

---

## 2. 详细结果

### 2.1 LIBERO-Plus — 按 suite（10,030 episodes）

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 2402 | 673 | 1729 | 28.02% |
| libero_object | 2518 | 757 | 1761 | 30.06% |
| libero_goal | 2591 | 659 | 1932 | 25.43% |
| libero_10 | 2519 | 445 | 2074 | 17.67% |
| **total** | **10030** | **2534** | **7496** | **25.26%** |

### 2.2 LIBERO-Plus — 按扰动类型（7 类，跨 4 个 suite 汇总）

| 扰动类型 | planned | successes | success_rate |
|---|---:|---:|---:|
| Background Textures（背景纹理） | 1076 | 481 | 44.70% |
| Objects Layout（物体布局） | 1525 | 817 | 53.57% |
| Language Instructions（语言指令） | 1537 | 566 | 36.82% |
| Light Conditions（光照条件） | 1142 | 218 | 19.09% |
| Robot Initial States（机器人初态） | 1550 | 214 | 13.81% |
| Sensor Noise（传感器噪声） | 1601 | 195 | 12.18% |
| **Camera Viewpoints（相机视角）** | 1599 | 43 | **2.69%** |

`Camera Viewpoints` 是 OpenVLA 最薄弱的扰动维度（2.69%），符合预期——单图输入策略对训练时未见过的相机视角变化天然缺乏鲁棒性，没有多视角/几何先验可以依赖。`Objects Layout`/`Background Textures` 相对最鲁棒（>44%），说明局部纹理和物体摆放的微小扰动尚可部分泛化。

### 2.3 LIBERO-Pro — 按 suite（8,000 episodes，`env` 扰动缺失，与 π0.5 口径一致）

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 2000 | 939 | 1061 | 46.95% |
| libero_object | 2000 | 835 | 1165 | 41.75% |
| libero_goal | 2000 | 641 | 1359 | 32.05% |
| libero_10 | 2000 | 292 | 1708 | 14.60% |
| **total** | **8000** | **2707** | **5293** | **33.84%** |

### 2.4 LIBERO-Pro — 按扰动类型（4 类，跨 4 个 suite 汇总）

| 扰动类型 | planned | successes | success_rate |
|---|---:|---:|---:|
| lan(semantic)（语言语义替换） | 2000 | 1310 | 65.50% |
| object（物体替换） | 2000 | 1124 | 56.20% |
| task（任务替换） | 2000 | 273 | 13.65% |
| **swap(position)（位置交换）** | 2000 | **0** | **0.00%** |

`swap(position)` 的 0.00% 已在 §1 核实为真实策略失败，详见 §4.2。`lan(semantic)`/`object` 两类扰动 OpenVLA 保留了较高鲁棒性（>56%），说明其失败集中在"空间/结构性"扰动而非"语言复述"扰动，与 §2.2 中 `Camera Viewpoints` 最弱的模式相互印证——OpenVLA 对空间几何关系的扰动明显比对纹理/语言表述的扰动更脆弱。

### 2.5 Smoke（44 episodes，流水线冒烟检查）

| Benchmark | planned | successes | success_rate |
|---|---:|---:|---:|
| LIBERO-Plus | 28 | 9 | 32.1% |
| LIBERO-Pro | 16 | 5 | 31.2% |

样本量小，仅用于验证流水线（checkpoint 加载、`--resume`、渲染写盘）可用，不具统计意义。

---

## 3. 评测配置

沿用 `../openvla-libero/EVAL_REPORT.md` §3 的 checkpoint/vision backbone/动作表示/`replan=1` 配置，额外说明：

| 项目 | 取值 |
|---|---|
| LIBERO-Pro `env` 扰动 | 4 个 suite 的 `env` cell 均为 `applicability=N/A`（`reason: "pre-generated cell is absent"`），与 π0.5 评测口径完全一致（预生成场景缺失，非本次评测遗漏），故 planned 分母为 8000 而非协议满额的 10000 |
| GPU 分配 | OpenVLA Plus/Pro 的 8 个 (suite × benchmark) 子任务分布在 GPU0-5 上并行执行，按实测吞吐动态调度（而非固定分配），期间为利用空闲 GPU 对 `plus-full-libero_10`/`plus-full-libero_object` 做过 episode 级分片加速，详见 §4.1 |
| 各 suite `max_steps` | `libero_spatial=220`、`libero_object=280`、`libero_goal=300`、`libero_10=520`，与 π0.5/`../openvla-libero` 一致 |

---

## 4. 注意事项 / 数据局限性

1. **episode 级 GPU 分片与合并**：为利用评测过程中空出的 GPU，`plus-full-libero_10`（3 路并行）与 `plus-full-libero_object`（2 路并行）曾被临时切分为多个并行子进程，各自负责不相交的 episode 子集（通过占位符 ledger 记录保证 `--resume` 逻辑不会重复调度）。全部子进程完成后，用自定义脚本 `merge_episode_shards.py`（复用 `eval_client._prepare` 的矩阵计算逻辑，而非重新实现）将各分片的真实记录合并回单一 `episodes.jsonl`，视频文件从符号链接"实体化"为常规文件后删除临时分片目录。合并结果经 `eval_support.verify_integrity()` 校验：`passed=true`，`issues=[]`，`missing_videos=0`，`records==planned`（`libero_10`: 2519/2519；`libero_object`: 2518/2518），不影响本报告任何统计口径。
2. **`swap(position)` 的 0.00% 是真实策略失败，已核实排除 bug 可能**：抽查全部 2000 条记录，`error_category` 均为 `null`，`action_steps` 均等于对应任务的 `max_steps`（100% 触发超时判负，无一提前崩溃/报错），与 §2.3/§2.4 讨论的"空间扰动脆弱、语言/物体扰动相对鲁棒"模式一致，判定为该架构（单图 + 无 chunk 逐步重规划）对训练时记忆的物体相对位置关系的依赖所致，不是评测流程 bug。
3. **评测过程中发生过两次 GPU 任务重复启动事故，均已定位修复，不影响最终数据**：一次在多套件驱动脚本按顺序自动续跑时与另一块已手动拉起的 GPU 冲突，在任何 episode 写入前即被发现并终止（zero impact）；另一次未及时发现，两个进程并发写入同一 `episodes.jsonl` 约 11 小时，产生 343 条真实重复记录——已备份原始文件（`.pre-dedup-backup`）后按 `episode_id+attempt` 去重、验证 0 条残留重复、重新以 `--resume` 方式续跑补全。两起事故均发生在 `pro-full-libero_spatial`/`pro-full-libero_object` 的调度阶段，最终数据已通过完整性校验，不影响本报告任何统计数字。
4. **零样本迁移，非同分布评测**：详见 §1，4 个 checkpoint 从未在 Plus/Pro 扰动数据上训练/微调过，本报告数字反映的是"标准 LIBERO 微调权重面对扰动数据时的分布外泛化能力"。
5. **smoke 样本量过小**：详见 §2.5，不具统计意义。
6. **全部 18,030 + 44 个 episode 均无基础设施错误**（`error_category` 全部为 `null`），失败均为真实的策略未达成任务目标。
7. 视频文件本地镜像未同步（仅同步 episodes/summary/manifests，体积原因），仍完整保留在远程 `/mnt/data/results/openvla-plus-pro/`，`episodes.jsonl` 中的 `video` 字段记录了原始路径。

---

## 5. 数据溯源

| 内容 | 路径 |
|---|---|
| 逐 episode 明细 | `plus-full-libero_{spatial,object,goal,10}/episodes.jsonl`、`pro-full-libero_{spatial,object,goal,10}/episodes.jsonl` |
| 成功率汇总表 | 各 run 目录下 `summary.csv` / `summary.json`（含 `groups.category`/`groups.perturbation` 细分） |
| 运行环境快照 | `<run>/manifests/environment-*.json` |
| Checkpoint 来源 | `<run>/manifests/checkpoint-server-*.json` |
| 分片合并记录 | 见 §4.1；合并脚本 `/tmp/merge_episode_shards.py`（临时脚本，未纳入仓库） |
| server/client 原始日志 | `<run>/logs/server-*.log`、`<run>/logs/client-egl-*.log` |
| 与 π0.5 / ACT 的横向对比 | `../analysis-report/cross_policy_report_zh.md` |

目录结构与字段含义详见同级 `../CLAUDE.md`。
