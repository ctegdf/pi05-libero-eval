# π0.5 (pi05_libero) × LIBERO-X Zero-Shot 评测记录

**评测对象**：`openpi` 仓库下的 π0.5 LIBERO 微调权重（`pi05_libero`，即标准 LIBERO 上的 official checkpoint，与 `../openpi-libero/EVAL_REPORT.md` 中 97.1% 成功率的同一份权重）
**评测基准**：[meituan/LIBERO-X](https://github.com/meituan/LIBERO-X)（vendor 快照 commit `f5287264`），LEVEL1–LEVEL5 全部五个难度级别
**数据来源**：远程主机 `<private>`，本地镜像目录 `libero-x/`
**评测代码版本**：`openpi` git commit `15a9616a`（与 `openpi-libero`/`plus-pro` 同一份工作区、同样的 8 处未提交本地补丁，详见 §4）
**评测时间**：2026-08-17 ~ 2026-08-24（约 7 天连续运行）

---

## 1. 结论速览

**这是一次零样本（zero-shot）迁移评测，不是 LIBERO-X 论文协议的复现**：`pi05_libero` 权重只在标准 LIBERO 数据上微调过，从未见过 LIBERO-X 新增的物体/场景/谓词；本次评测把它直接拿到 LIBERO-X 的 5 个 LEVEL 上跑，用于衡量微调权重的分布外泛化能力，而非公平对比论文中"在 LIBERO-X 上微调"的结果。

| LEVEL | Planned | Successes | Failures | Success rate |
|---|---:|---:|---:|---:|
| LEVEL1 | 6000 | 458 | 5542 | 7.6% |
| LEVEL2 | 6000 | 419 | 5581 | 7.0% |
| LEVEL3 | 6000 | 478 | 5522 | 8.0% |
| LEVEL4 | 8260 | 703 | 7557 | 8.5% |
| LEVEL5 | 8260 | 653 | 7607 | 7.9% |
| **合计** | **34,520** | **2711** | **27,809** | **7.9%** |

**核心结论**：与标准 LIBERO 上 97.1% 的成功率相比，同一份权重在 LIBERO-X 上跌到约 7.9%，五个 LEVEL 之间的差异很小（7.0%–8.5%，标准差约 0.5 个百分点），说明性能下降主要来自"新物体/新场景/新谓词"这一整体分布偏移，而非某个特定 LEVEL 的难度更高——微调权重对 LIBERO-X 引入的新颖性（新物体外观/材质、新增 `ExactIn`/`UprightOn`/`SideOn` 谓词、新场景布局）几乎没有泛化能力。全部 34,520 个 episode 均在 `max_steps=1200` 内正常运行完毕、渲染视频写盘成功，**0 例基础设施错误**（连接/环境/checkpoint/policy_runtime 类错误计数均为 0），失败均为真实的策略未达成任务目标。

---

## 2. 详细结果

### 2.1 Full（34,520 episodes，全部 5 个 LEVEL）

| LEVEL | 任务结构 | planned | successes | failures | success_rate |
|---|---|---:|---:|---:|---:|
| LEVEL1 | 600 tasks × 10 trials | 6000 | 458 | 5542 | 7.63% |
| LEVEL2 | 600 tasks × 10 trials | 6000 | 419 | 5581 | 6.98% |
| LEVEL3 | 600 tasks × 10 trials | 6000 | 478 | 5522 | 7.97% |
| LEVEL4 | 826 tasks × 10 trials | 8260 | 703 | 7557 | 8.51% |
| LEVEL5 | 826 tasks × 10 trials（同 LEVEL4 场景，5 组同义 instruction 按 trial 轮换） | 8260 | 653 | 7607 | 7.91% |
| **total** | | **34,520** | **2711** | **27,809** | **7.85%** |

LEVEL5 的 5 组语言复述变体（`L5-1`~`L5-5`，每组各 1652 episode，按 trial index 轮换分配，而非独立倍增样本量）成功率彼此接近，未出现某种复述方式明显更易/更难：

| 复述变体 | successes | planned | success_rate |
|---|---:|---:|---:|
| L5-1 | 124 | 1652 | 7.5% |
| L5-2 | 138 | 1652 | 8.4% |
| L5-3 | 133 | 1652 | 8.1% |
| L5-4 | 141 | 1652 | 8.5% |
| L5-5 | 117 | 1652 | 7.1% |

说明策略对 LIBERO-X 的失败**不是**语言理解/指令措辞层面的问题，而是视觉/操作层面的分布外泛化失败（新物体、新场景布局识别与抓取失败）。

各 LEVEL 的实际墙钟运行区间（含全部 5-6 GPU 并行分片，跨 7 天连续运行，中途未重启进程、无崩溃）：

| LEVEL | GPU | started_at | finished_at | 时长 |
|---|---|---|---|---|
| LEVEL1 | 0 | 2026-08-17T13:36:07Z | 2026-08-22T06:22:19Z | ~4.7 天 |
| LEVEL2 | 6 | 2026-08-17T14:02:47Z | 2026-08-22T07:50:39Z | ~4.7 天 |
| LEVEL3 | 1 | 2026-08-17T13:38:58Z | 2026-08-22T13:05:31Z | ~5.0 天 |
| LEVEL4 | 2 | 2026-08-17T13:40:16Z | 2026-08-24T08:05:25Z | ~6.8 天 |
| LEVEL5 | 3 | 2026-08-17T13:41:46Z | 2026-08-24T09:09:10Z | ~6.8 天 |

LEVEL2 最初与 LEVEL1 共用同一分片（GPU0），运行过程中按用户指示拆分并迁移到 GPU6 独立运行，以缓解 GPU0 过载；迁移过程通过精确 PID 的 SIGTERM 优雅停止、按已完成的 episode 记录分裂 ledger 后以 `--resume` 方式在两块 GPU 上继续，未产生数据丢失或重复（详见 §4 与本次评测使用的 resume-safe 归档机制）。

### 2.2 Smoke（29 episodes，流水线冒烟检查）

| LEVEL | planned | successes | failures | success_rate |
|---|---:|---:|---:|---:|
| LEVEL1 | 6 | 1 | 5 | 16.7% |
| LEVEL2 | 6 | 1 | 5 | 16.7% |
| LEVEL3 | 6 | 0 | 6 | 0.0% |
| LEVEL4 | 6 | 0 | 6 | 0.0% |
| LEVEL5 | 5 | 0 | 5 | 0.0% |
| **total** | **29** | **2** | **27** | **6.9%** |

用途：正式跑 full 前验证流水线（checkpoint 加载、`--resume` 断点续跑、渲染写盘、`libero_x_support`/`libero_x_client` 各阶段）是否可用，样本量小，不具统计意义。最终 full 结果（7.85%）与 smoke 结果（6.9%）量级一致，验证了 smoke 阶段的代表性。

### 2.3 Preflight

`server_startup` 与单条 `client_run` 均 `passed`（`exit_code=0`），验证了 checkpoint 加载、LIBERO-X BDDL/init 文件解析、`OffScreenRenderEnv` 初始化、policy server 通信链路全部可用，未产生正式 episode 记录。

---

## 3. 评测配置

| 项目 | 取值 |
|---|---|
| 评测协议名 | `pi05-libero-zero-shot`（自定义命名，强调这是零样本迁移评测，非论文复现协议） |
| 代码仓库 | `openpi`（π0.5 官方仓库） + 本次新增的 `pi05-libero-x-eval` 评测脚手架 |
| Git HEAD（openpi） | `15a9616a00943ada6c20a0f158e3adb39df2ccac`，工作区带与 `openpi-libero`/`plus-pro` 相同的本地补丁（详见 §4） |
| LIBERO-X vendor 快照 | commit `f5287264`（通过 GitHub tree API 逐文件下载获取，非 `git clone`，vendor 目录本身不含 `.git`，完整性以逐文件 SHA1 抽样核对） |
| 微调权重 checkpoint | `pi05_libero`（`config=pi05_libero`, `protocol=official`），与 `openpi-libero/official-full` 使用的是**同一份**已验证 checkpoint |
| norm_stats | `physical-intelligence/libero` asset（sha256 `b3a44bb2...`），与标准 LIBERO 评测共用 |
| 渲染/图像分辨率 | `resize=224` |
| 重规划间隔 | `replan=5` |
| `max_steps` | 1200（固定值，取自 LIBERO-X 官方 `eval_template.py` 的默认协议，**不是**论文里各任务各异的 per-task horizon；见 §4 局限性 1） |
| `executed_wait_steps` | 0（严格对齐官方模板，不额外注入等待步） |
| 环境初始化 | `env.seed(trial)` + `regenerate_obs_from_state(init_state)`，逐 trial 复位，与官方模板一致 |
| GL 后端 | `egl`（headless） |
| GPU 分配 | LEVEL1→GPU0，LEVEL2→GPU6（中途从 GPU0 迁出），LEVEL3→GPU1，LEVEL4→GPU2，LEVEL5→GPU3 |
| Python 环境 | 新建 `liberox` conda env（Python 3.9），CPU-only torch 1.11.0（仅需 `torch.load` 读取 init states，GPU 推理由已验证的 openpi `.venv` server 进程承担） |

各 LEVEL 的任务规模：LEVEL1–3 均为 600 tasks × 10 trials = 6000；LEVEL4/5 均为 826 tasks × 10 trials = 8260（LEVEL5 复用 LEVEL4 的 BDDL 场景，额外附加 5 组同义 instruction，按 `(trial // 2) % 5` 规则轮换分配到 10 个 trial 中，而非独立倍增）。

---

## 4. 注意事项 / 数据局限性

1. **`max_steps=1200` 是固定值，非论文的 per-task horizon 协议**：本次评测遵循 LIBERO-X 官方 `eval_template.py` 默认参数（全任务统一 1200 步上限），这是"按官方评测脚本跑一遍"的协议，不是论文中报告数字所用的确切协议（如论文另有逐任务时限设定，本报告数字与论文数字不可直接对比，只能作为同一份 checkpoint 的自评基线）。
2. **零样本迁移，非同分布评测**：`pi05_libero` 从未在 LIBERO-X 数据上训练/微调过。7.9% 的整体成功率反映的是"标准 LIBERO 微调权重面对 LIBERO-X 新增物体/场景/谓词时的分布外泛化能力"，不是"LIBERO-X 训练后的模型能力上限"。不要将其与 LIBERO-X 论文中（若有）微调后的结果相比较。
3. **工作区非纯净提交**：`openpi` 侧 git 状态与 `openpi-libero`/`plus-pro` 完全一致（`git-openpi-*.json` 显示相同的 8 处改动：`M scripts/train.py`、`M src/openpi/models/model.py`、`M src/openpi/training/config.py`、`D pyproject.toml`、`D uv.lock`，以及新增的 `examples/libero/pi05_libero_{client,server,support}.py`、`run_pi05_libero.py`、`test_pi05_libero_eval.py`、`start_server.sh`、`examples/franka_isaaclab/`），这些是既有 LIBERO 评测脚手架的本地补丁，policy server 侧代码与其余两个评测系列完全共用同一份逻辑，不影响可比性。
4. **LIBERO-X vendor 目录不是 git checkout**：因 GitHub 下载带宽受限，改用 GitHub tree API 逐文件通过 `raw.githubusercontent.com` 并行拉取（而非 `git clone`），故 `git-vendor-*.json` 中 `commit` 为 `null`（"not a git repository"）；实际快照版本 `f5287264` 记录在 harness 代码注释中，完整性通过全量文件大小比对 + 60 个文件的随机 SHA1 抽样核对（0 mismatch）确认，不影响评测结果的可信度，仅影响可用标准 git 工具直接溯源。
5. **LEVEL2 中途迁移 GPU**：详见 §2.1，迁移使用 resume-safe 机制（按 episode_id 精确续跑、无重复无丢失），迁移前后的 episode 记录已合并进最终的 `full/level2/episodes.jsonl`，对最终统计无影响。
6. **`cross_check_registry` 仅覆盖 LEVEL1–3**：LIBERO-X 官方 benchmark registry（`libero.libero_x.benchmark`）要求特定的 `SCENE_LEVEL<n>` 子目录结构，与本次评测使用的 vendor 路径布局不完全匹配，故该项校验对 LEVEL4/LEVEL5 会被跳过（`status=skipped`），仅作为可选的双重核对，不影响任务发现/expand_matrix 主流程（后者基于文件系统直接扫描 BDDL/init 文件，已独立验证 episode 总数与规划值完全一致）。
7. **smoke/full 均已通过完整性校验**：每个 LEVEL 的 `summary.json.integrity` 字段均为 `{"passed": true, "issues": [], "missing_videos": 0}`，且 `records == planned`，无缺失/重复 episode。
8. **全部 34,520 + 29 个 episode 均有对应 `.mp4` 渲染视频落盘**，路径见各 `videos/` 目录及 `episodes.jsonl` 中的 `video` 字段，可用于人工复核个别失败案例（例如判断失败是"接近成功的微小误差"还是"完全找错物体/方向"）。

---

## 5. 数据溯源

| 内容 | 路径 |
|---|---|
| 逐 episode 明细 | `full/level{1..5}/episodes.jsonl`、`smoke/episodes.jsonl` |
| 成功率汇总表 | 各 run 目录下 `summary.csv` / `summary.json`（含 `groups.level`、`groups.prompt_field` 细分） |
| 运行环境快照 | `<run>/manifests/environment-*.json` |
| 代码版本/工作区状态 | `<run>/manifests/git-openpi-*.json`、`git-vendor-*.json` |
| Checkpoint 来源 | `<run>/manifests/checkpoint-server-*.json` |
| 流水线阶段时间线 | `<run>/manifests/result-*.json`（`events` 数组） |
| server/client 原始日志 | `<run>/logs/server-*.log`、`<run>/logs/client-egl-*.log` |
| 渲染视频 | `<run>/videos/*.mp4` |
| 评测代码 | `harness/{libero_x_support,pi05_liberox_client,pi05_liberox_server,run_pi05_liberox_eval}.py` |

目录结构与字段含义详见同级 `../CLAUDE.md`。与标准 LIBERO（97.1%）、LIBERO-Pro/Plus 结果的横向对比参见 `../openpi-libero/EVAL_REPORT.md` 与 `../plus-pro/`（如已产出报告）。
