# π0.5 vs OpenVLA vs ACT 跨策略对比报告

**对比对象**：3 个策略在同一套 LIBERO / LIBERO-Plus / LIBERO-Pro 评测协议下的表现
- **π0.5**（`pi05_libero`）：单一统一权重，覆盖全部 4 个 LIBERO 子集，chunk 式动作预测（`replan=5`）
- **OpenVLA-7B**（`openvla-7b-finetuned-libero-*`）：4 个按子集独立微调的官方权重，逐步重规划（`replan=1`，无 chunk）
- **ACT**：4 个按子集从头训练的 `lerobot` ACT 策略，无语言条件，动作 chunk 更大（`chunk_size=100`）

**数据口径**：均取自各自的 full 阶段结果（π0.5 为已有归档 `openpi-libero/official-full`+`plus-pro/plus-full-merged`+`plus-pro/pro-full`；OpenVLA/ACT 为 4 个 per-suite 目录按 `episode_id` 去重最新 attempt 后汇总），全部数字由 `generate_cross_policy_report.py` 从本地镜像的 `episodes.jsonl` 直接计算得出，未手工调整。三个策略在全部 3×8000~10030 episode 规模上均**零基础设施错误**（`error_category` 全部为 `null`）。

**重要口径提示**：这不是一次架构公平对比——三个策略在 checkpoint 数量（π0.5 单权重 vs OpenVLA/ACT 各 4 权重）、动作预测节奏（`replan=1/5/100`）、是否接入语言条件（ACT 不接入）、模型规模（OpenVLA 7B vs π0.5/ACT 量级不同）上均有本质差异。本报告呈现的是"三套已发布/已训练权重按各自官方协议评测"的横向数字，用于了解相对差距和共性弱点，不用于论证某种架构本身更优。

---

## 1. 总览：三个 benchmark × 三个策略

| Benchmark | π0.5 | OpenVLA | ACT | 规模 |
|---|---:|---:|---:|---:|
| LIBERO（标准四套件，同分布） | **97.10%** | 69.15%（CI95 [67.1%,71.1%]） | 24.90%（CI95 [23.1%,26.8%]） | 2,000 episodes |
| LIBERO-Plus（零样本） | **83.65%** | 25.26%（CI95 [24.4%,26.1%]） | 13.06%（CI95 [12.4%,13.7%]） | 10,030 episodes |
| LIBERO-Pro（零样本，8000/10000，`env` 扰动缺失） | **58.79%** | 33.84%（CI95 [32.8%,34.9%]） | 11.11%（CI95 [10.4%,11.8%]） | 8,000 episodes |

**排名在三个 benchmark 上完全一致**：π0.5 ≫ OpenVLA > ACT，没有出现排名反转。三者的相对差距在扰动引入后进一步拉大——同分布下 π0.5 领先 OpenVLA 约 28pp，Plus/Pro 上领先约 58pp/25pp（Pro 因两者降幅接近，差距反而略微收窄，见 §2）。

### 1.1 从同分布到零样本迁移的降幅（绝对百分点）

| 策略 | 同分布基线 | → Plus | → Pro |
|---|---:|---:|---:|
| π0.5 | 97.10% | **−13.5pp**（→83.65%） | **−38.3pp**（→58.79%） |
| OpenVLA | 69.15% | **−43.9pp**（→25.26%） | **−35.3pp**（→33.84%） |
| ACT | 24.90% | **−11.8pp**（→13.06%） | **−13.8pp**（→11.11%） |

**关键发现——两种截然不同的"脆弱模式"**：
- **π0.5**：对 LIBERO-Plus 的视觉/环境扰动（背景纹理、光照、相机视角等）高度稳健（仅 −13.5pp），但对 LIBERO-Pro 的结构性扰动（物体位置交换、任务替换）明显更脆弱（−38.3pp）——说明其鲁棒性来自对"表观变化"的泛化能力，而非对任务结构本身的深层理解。
- **OpenVLA**：模式相反，对 Plus 的降幅（−43.9pp）反而大于 Pro（−35.3pp）——在 Pro 上两者降幅已经相当接近（−38.3pp vs −35.3pp），说明结构性扰动对 π0.5 和 OpenVLA 造成了相近量级的冲击，π0.5 的整体优势主要建立在 Plus 维度的表观鲁棒性上，而非在"真正理解任务结构"这件事上有数量级的领先。
- **ACT**：两个 benchmark 的降幅都不大（−11.8pp / −13.8pp），但这是因为它的同分布基线本身就低（24.90%），可跌的空间有限，不代表它比另外两者更"稳健"。

---

## 2. LIBERO-Plus：按扰动类型的三方对比（7 类，跨 4 个 suite 汇总）

| 扰动类型 | π0.5 | OpenVLA | ACT |
|---|---:|---:|---:|
| Background Textures（背景纹理） | 94.05% | 44.70% | 13.94% |
| Light Conditions（光照条件） | 94.22% | 19.09% | 18.56% |
| Sensor Noise（传感器噪声） | 86.95% | 12.18% | 4.37% |
| Language Instructions（语言指令） | 85.88% | 36.82% | 23.23%* |
| Objects Layout（物体布局） | 84.98% | 53.57% | 13.38% |
| Robot Initial States（机器人初态） | 73.16% | 13.81% | 15.48% |
| Camera Viewpoints（相机视角） | 72.55% | **2.69%** | **4.82%** |

\* ACT 在 `Language Instructions` 类别下的分数**不代表语言鲁棒性**——ACT 本身不接入语言条件，该类别的分数实质是其在对应场景视觉配置下的基线表现，详见 `../act-plus-pro/EVAL_REPORT.md` §1/§4.1。

**共性弱点**：`Camera Viewpoints`（相机视角）是 OpenVLA 和 ACT **共同**的最弱扰动维度（分别 2.69%/4.82%），π0.5 在此维度上虽也是其 7 类中偏弱的一档（72.55%），但降幅远小于另外两者——三个策略的相对排序在几乎每个扰动维度上都一致（π0.5 > OpenVLA > ACT），只有 `Robot Initial States` 上 ACT 略高于 OpenVLA（15.48% vs 13.81%），`Light Conditions` 上两者基本持平（18.56% vs 19.09%），其余维度 OpenVLA 均明显领先 ACT。

---

## 3. LIBERO-Pro：按扰动类型的三方对比（4 类，跨 4 个 suite 汇总）

| 扰动类型 | π0.5 | OpenVLA | ACT |
|---|---:|---:|---:|
| lan(semantic)（语言语义替换） | 96.30% | 65.50% | 24.50%* |
| object（物体替换） | 86.45% | 56.20% | 18.65% |
| swap(position)（位置交换） | 28.10% | **0.00%** | **0.35%** |
| task（任务替换） | 24.30% | 13.65% | 0.95% |

\* 同 §2 注：ACT 在 `lan(semantic)` 下的分数不代表语言鲁棒性。

**最突出的单点发现——`swap(position)`（物体位置交换）扰动对三个策略都是最严峻的考验，且呈现明显的架构分野**：
- π0.5（chunk 式动作预测 + 双图输入）：28.10%，三者中唯一还保有两位数成功率的策略。
- ACT（大 chunk，但同样无语言条件、模仿学习）：0.35%（7/2000），几近全灭但未到 0。
- **OpenVLA（单图输入、逐步重规划、无 chunk）：0.00%（0/2000），全部 2000 个 episode 无一成功**，已在 `../openvla-plus-pro/EVAL_REPORT.md` §4.2 核实排除基础设施错误的可能，是真实的策略盲区。

`task`（任务替换）是 π0.5/OpenVLA/ACT 共同的第二薄弱扰动（24.30%/13.65%/0.95%），说明"重新组合已知子任务成新任务"这类结构性泛化，对三种架构都比"替换物体外观"或"替换指令措辞"更难，且难度随模型能力递减而急剧放大（π0.5 24.30% → OpenVLA 13.65% → ACT 0.95%，接近指数式衰减而非线性衰减）。

---

## 4. LIBERO：按 suite 的三方对比（标准四套件，同分布）

| suite | π0.5 | OpenVLA | ACT |
|---|---:|---:|---:|
| libero_spatial | 98.6% | 76.6% | 38.8% |
| libero_object | 98.4% | 87.0% | 35.2% |
| libero_goal | 98.2% | 72.2% | **3.0%** |
| libero_10 | 93.2% | 40.8% | 22.6% |

`libero_goal` 是唯一一个三个策略排序**不**保持"π0.5 > OpenVLA > ACT"平滑梯度的 suite——ACT 在此 suite 上断崖式下跌（3.0%，远低于其自身在其余 3 个 suite 上的 22.6%~38.8%），原因是 §2/§3 反复提到的"无语言条件"架构限制：`libero_goal` 套件的任务区分度主要来自指令文本而非视觉场景差异，ACT 结构上无法利用这一信息通道，故这是已知架构限制而非训练/评测异常（详见 `../act-libero-full/EVAL_REPORT.md` §4.2）。`libero_10`（长程组合任务）在 π0.5/OpenVLA 两者上都是各自最弱的 suite（93.2%/40.8%），符合"长程任务对误差累积更敏感"的一般预期。

---

## 5. 结论与解读边界

1. **一致的能力排序**：π0.5 ≫ OpenVLA > ACT，在同分布 LIBERO、LIBERO-Plus、LIBERO-Pro 三个 benchmark、以及几乎每一个扰动维度/suite 细分下都成立，唯一的部分例外是 §4 讨论的 `libero_goal`（ACT 因无语言条件而额外塌陷）。
2. **π0.5 的鲁棒性优势主要来自"表观扰动"维度，而非"结构性扰动"维度**：见 §1.1，这是本报告最重要的细粒度发现——粗看"π0.5 大幅领先"的结论，拆解到 Plus vs Pro 两个维度后，Pro 上 π0.5 和 OpenVLA 的降幅其实接近，领先优势明显收窄。
3. **`swap(position)`/`Camera Viewpoints`/`task` 是三个策略（尤其两个模仿学习基线 OpenVLA/ACT）的共同薄弱环节**，可能提示的是训练数据本身在这些维度上覆盖不足，而非某一具体架构的特有缺陷，值得作为后续改进方向的共性线索。
4. **不是架构公平对比**：详见页首"重要口径提示"，checkpoint 数量、动作节奏、语言条件、模型规模均不同，不能把本报告的数字直接归因为"某种架构设计选择本身更优"。
5. **全部对比数字均来自零基础设施错误的干净数据**（9 个 policy×benchmark 组合、共计约 42,000+ episode，`error_category` 全部为 `null`），失败均为真实的策略未达成任务目标，不存在评测流程干扰对比结论的情况。

---

## 6. 数据来源与复现

| 内容 | 路径 |
|---|---|
| 生成脚本 | `generate_cross_policy_report.py`（本目录，导入 `generate_report.py` 的 `read_jsonl`/`aggregate`/`grouped`/`wilson`/`LABELS` 等工具函数，不修改该脚本本体） |
| 中间数据 CSV | `data/cross_policy_total.csv`、`data/cross_policy_by_suite.csv`、`data/cross_policy_by_category.csv`、`data/cross_policy_by_perturbation.csv` |
| π0.5 原始数据 | `../openpi-libero/official-full/`、`../plus-pro/plus-full-merged/`、`../plus-pro/pro-full/` |
| OpenVLA 原始数据 | `../openvla-libero/`、`../openvla-plus-pro/`（各自的 `EVAL_REPORT.md` 有更详细的单策略分析） |
| ACT 原始数据 | `../act-libero-full/`、`../act-plus-pro/`（各自的 `EVAL_REPORT.md` 有更详细的单策略分析，含夹爪 bug 修复历史） |

复现方式：`cd analysis-report && python3 generate_cross_policy_report.py`（需要本地镜像已同步对应 campaign 的 `episodes.jsonl`，视频文件不参与本报告计算，无需同步）。
