# ACT（按子集从头训练）× LIBERO-Plus / LIBERO-Pro Zero-Shot 评测记录

**评测对象**：4 个 ACT 策略（与 `../act-libero-full/EVAL_REPORT.md` 同一批，`lerobot` 实现，各自按 LIBERO 子集官方训练集从头训练，无语言条件）
**评测基准**：LIBERO-Plus（7 类扰动 × 5 档难度）+ LIBERO-Pro（4 类结构性扰动；`env` 扰动 cell 缺失，与 π0.5/OpenVLA 评测口径一致）
**数据来源**：远程主机 `<private>`，本地镜像目录 `act-plus-pro/`
**评测代码版本**：`pi05-libero-plus-pro-eval` harness，同 `../act-libero-full/EVAL_REPORT.md`
**评测时间**：2026-08-27 ~ 2026-09-01（约 4.9 天，Plus/Pro 两个 benchmark 分别耗时约 3.7 天 / 1.9 天，部分重叠并行）

---

## 1. 结论速览

**零样本迁移评测**：4 个 checkpoint 从未在 Plus/Pro 扰动数据上训练过。

| Benchmark | Planned/Attempted | Successes | Failures | Success rate | 对照：π0.5 | 对照：OpenVLA | 对照：ACT 同分布 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LIBERO-Plus | 10,030 | 1,310 | 8,720 | **13.06%**（CI95 [12.4%, 13.7%]） | 83.65% | 25.26% | 24.90% |
| LIBERO-Pro | 8,000 / 10,000 | 889 | 7,111 | **11.11%**（CI95 [10.4%, 11.8%]） | 58.79% | 33.84% | 24.90% |

**核心结论**：ACT 是三个策略中零样本迁移能力最弱的——LIBERO-Plus 从同分布的 24.90%（`../act-libero-full`）跌到 13.06%（−11.8pp），LIBERO-Pro 跌到 11.11%（−13.8pp），降幅本身小于 OpenVLA（因为 ACT 同分布基线已经不高，可跌的空间有限），但绝对水平在三者中垫底，全面弱于 π0.5 和 OpenVLA。一个值得注意的反常现象：ACT 在 Plus 的 `Language Instructions`（语言指令）扰动类别上反而是它表现**最好**的类别（23.23%，高于其余 6 类），Pro 的 `lan(semantic)`（语言语义替换）扰动上也是它表现最好的扰动类型（24.50%）——这**不是** ACT 对语言扰动特别鲁棒，而是因为 §3 所述 ACT 本身**不接入语言条件**，语言层面的扰动对它而言完全不可见，这一类别下的分数实质反映的是其在该场景视觉配置下的基线表现，与是否"理解"了扰动无关，解读时不应误读为"语言鲁棒性"。全部 18,030 个 episode 均无基础设施错误。

---

## 2. 详细结果

### 2.1 LIBERO-Plus — 按 suite（10,030 episodes）

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 2402 | 381 | 2021 | 15.86% |
| libero_object | 2518 | 566 | 1952 | 22.48% |
| libero_goal | 2591 | 81 | 2510 | 3.13% |
| libero_10 | 2519 | 282 | 2237 | 11.19% |
| **total** | **10030** | **1310** | **8720** | **13.06%** |

`libero_goal` 再次是 ACT 最弱的 suite（3.13%），与 `../act-libero-full` 同分布结果（3.0%）几乎一致——符合预期，无语言条件的架构限制在 Plus 扰动下同样成立，不因加入视觉扰动而改变。

### 2.2 LIBERO-Plus — 按扰动类型（7 类，跨 4 个 suite 汇总）

| 扰动类型 | planned | successes | success_rate |
|---|---:|---:|---:|
| **Language Instructions（语言指令）** | 1537 | 357 | **23.23%**（见 §1 解读说明，非真实语言鲁棒性） |
| Light Conditions（光照条件） | 1142 | 212 | 18.56% |
| Robot Initial States（机器人初态） | 1550 | 240 | 15.48% |
| Background Textures（背景纹理） | 1076 | 150 | 13.94% |
| Objects Layout（物体布局） | 1525 | 204 | 13.38% |
| Camera Viewpoints（相机视角） | 1599 | 77 | 4.82% |
| Sensor Noise（传感器噪声） | 1601 | 70 | 4.37% |

`Camera Viewpoints`/`Sensor Noise` 是 ACT 最薄弱的两类（均 <5%），与 OpenVLA 在 Plus 上同样最弱的 `Camera Viewpoints`（2.69%，见 `../openvla-plus-pro/EVAL_REPORT.md` §2.2）模式一致——视角/传感器层面的分布外扰动对两个策略都是共同弱点，可能反映的是共性问题（训练数据视角覆盖不足），而非某一架构特有的缺陷。

### 2.3 LIBERO-Pro — 按 suite（8,000 episodes）

| suite | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| libero_spatial | 2000 | 350 | 1650 | 17.50% |
| libero_object | 2000 | 349 | 1651 | 17.45% |
| libero_goal | 2000 | 39 | 1961 | 1.95% |
| libero_10 | 2000 | 151 | 1849 | 7.55% |
| **total** | **8000** | **889** | **7111** | **11.11%** |

### 2.4 LIBERO-Pro — 按扰动类型（4 类，跨 4 个 suite 汇总）

| 扰动类型 | planned | successes | success_rate |
|---|---:|---:|---:|
| lan(semantic)（语言语义替换） | 2000 | 490 | 24.50%（见 §1，非真实语言鲁棒性） |
| object（物体替换） | 2000 | 373 | 18.65% |
| task（任务替换） | 2000 | 19 | 0.95% |
| **swap(position)（位置交换）** | 2000 | 7 | **0.35%** |

`swap(position)` 同样是 ACT 最弱的扰动类型（0.35%），与 OpenVLA 在此扰动上的 0.00%（`../openvla-plus-pro/EVAL_REPORT.md` §2.4）、π0.5 的 28.10%（`../analysis-report/cross_policy_report_zh.md`）相互印证：位置交换扰动是三个策略（尤其两个模仿学习基线）共同的薄弱环节，π0.5 因 chunk 式规划 + 更强的空间表征相对更鲁棒。

### 2.5 Smoke（44 episodes，流水线冒烟检查）

| Benchmark | planned | successes | success_rate |
|---|---:|---:|---:|
| LIBERO-Plus | 28 | 6 | 21.4% |
| LIBERO-Pro | 16 | 1 | 6.2% |

样本量小，不具统计意义，仅用于流水线冒烟检查。

---

## 3. 评测配置

沿用 `../act-libero-full/EVAL_REPORT.md` §3 的训练数据/模型实现/`chunk_size=100`/**无语言条件**配置。LIBERO-Pro `env` 扰动 cell 缺失（`applicability=N/A`）与 π0.5/OpenVLA 口径一致，planned 分母为 8000。

---

## 4. 注意事项 / 数据局限性

1. **"Language Instructions"/"lan(semantic)" 类别下的相对高分不代表语言鲁棒性**：详见 §1/§2.2/§2.4，ACT 本身不接入语言指令（`../act-libero-full/EVAL_REPORT.md` §4.2 所述架构限制在 Plus/Pro 上同样成立），这两个类别下的分数实质是"该场景视觉配置下的基线表现"，解读跨模型对比时需特别注意这一点，不应与 OpenVLA/π0.5 在同一维度下的分数做字面意义上的"语言鲁棒性"对比。
2. **`libero_goal` 持续是最弱 suite**（Plus 3.13% / Pro 1.95%），与同分布结果（3.0%）一致，进一步印证 §1 的架构限制而非扰动引入的新问题。
3. **零样本迁移，非同分布评测**：4 个 checkpoint 从未在 Plus/Pro 数据上训练过。
4. **`swap(position)`/`Camera Viewpoints`/`Sensor Noise` 是 ACT 与 OpenVLA 的共同薄弱环节**：详见 §2.2/§2.4 讨论，可能反映训练数据本身在视角覆盖和空间扰动鲁棒性上的共性局限，而非单一架构特有缺陷。
5. **smoke 样本量过小**：详见 §2.5，不具统计意义。
6. **全部 18,030 + 44 个 episode 均无基础设施错误**（`error_category` 全部为 `null`），失败均为真实的策略未达成任务目标。
7. 视频文件本地镜像未同步，仍完整保留在远程 `/mnt/data/results/act-plus-pro/`。

---

## 5. 数据溯源

| 内容 | 路径 |
|---|---|
| 逐 episode 明细 | `plus-full-libero_{spatial,object,goal,10}/episodes.jsonl`、`pro-full-libero_{spatial,object,goal,10}/episodes.jsonl` |
| 成功率汇总表 | 各 run 目录下 `summary.csv` / `summary.json`（含 `groups.category`/`groups.perturbation` 细分） |
| 运行环境快照 | `<run>/manifests/environment-*.json` |
| Checkpoint 来源 | `<run>/manifests/checkpoint-server-*.json` |
| server/client 原始日志 | `<run>/logs/server-*.log`、`<run>/logs/client-egl-*.log` |
| 与 π0.5 / OpenVLA 的横向对比 | `../analysis-report/cross_policy_report_zh.md` |

目录结构与字段含义详见同级 `../CLAUDE.md`。
