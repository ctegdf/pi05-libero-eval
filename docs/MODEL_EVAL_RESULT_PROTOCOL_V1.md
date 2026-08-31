# Model Evaluation Result Protocol v1

## 1. Purpose

本协议用于统一记录 ACT、OpenVLA 以及后续模型在 LIBERO、LIBERO-Plus、LIBERO-Pro 上的评测结果。

本协议只规定：

- 结果目录结构
- 运行元数据格式
- episode 结果格式
- 汇总格式
- 视频路径格式
- 断点续跑规则
- 验收标准

本协议不规定：

- 模型训练方法
- checkpoint 加载方式
- 推理框架
- GPU 分配方式
- benchmark 启动命令
- ACT/OpenVLA 的具体实现细节

## 2. Fixed Identifiers

### Model IDs

模型 ID 必须使用以下小写值：

```text
act
openvla
```

展示名称可以使用：

```text
ACT
OpenVLA
```

### Benchmark IDs

```text
libero
libero-plus
libero-pro
```

### Evaluation Phases

```text
smoke
full
```

### Run Status

```text
planned
running
completed
partial
failed
cancelled
```

## 3. Directory Layout

所有新结果必须放在：

```text
results/runs/<model_id>/<benchmark_id>/<run_id>/
```

示例：

```text
results/runs/act/libero/act-libero-full-v1/
results/runs/act/libero-plus/act-libero-plus-full-v1/
results/runs/act/libero-pro/act-libero-pro-full-v1/

results/runs/openvla/libero/openvla-libero-full-v1/
results/runs/openvla/libero-plus/openvla-libero-plus-full-v1/
results/runs/openvla/libero-pro/openvla-libero-pro-full-v1/
```

每个 run 目录必须包含：

```text
run.json
episodes.jsonl
summary.json
summary.csv
```

如果保存视频，视频不得放入 GitHub 仓库，统一放入独立视频目录：

```text
videos/<model_id>/<benchmark_id>/<run_id>/
```

例如：

```text
videos/act/libero/act-libero-full-v1/
videos/openvla/libero-plus/openvla-libero-plus-full-v1/
```

## 4. Run ID Rules

`run_id` 必须：

- 全局唯一
- 只使用 ASCII 字符
- 使用小写字母、数字和连字符
- 不包含绝对路径
- 不包含 token、用户名或服务器地址

推荐格式：

```text
<model_id>-<benchmark_id>-<phase>-<version>
```

示例：

```text
act-libero-smoke-v1
act-libero-full-v1
openvla-libero-plus-full-v1
```

如果同一配置重复运行，增加 attempt：

```text
act-libero-full-v1-attempt-02
```

## 5. `run.json` Protocol

每个 run 必须有一个 `run.json`。

标准模板：

```json
{
  "schema_version": 1,
  "run_id": "act-libero-full-v1",
  "model_id": "act",
  "model_name": "ACT",
  "benchmark": "libero",
  "phase": "full",
  "status": "planned",

  "checkpoint": {
    "name": null,
    "revision": null,
    "sha256": null
  },

  "protocol": {
    "seed": 7,
    "trial_count": null,
    "suite_selection": null,
    "task_selection": null,
    "max_steps_policy": null,
    "action_horizon": null
  },

  "implementation": {
    "inference_framework": null,
    "repository": null,
    "repository_revision": null,
    "config_file": null
  },

  "provenance": {
    "created_at": null,
    "started_at": null,
    "finished_at": null,
    "hardware": null,
    "cuda_version": null,
    "python_version": null
  },

  "counts": {
    "expected_episodes": null,
    "recorded_episodes": 0,
    "successes": 0,
    "failures": 0,
    "errors": 0,
    "videos_available": 0,
    "videos_missing": 0
  },

  "artifacts": {
    "episodes": "episodes.jsonl",
    "summary_json": "summary.json",
    "summary_csv": "summary.csv",
    "video_root": null
  },

  "notes": null
}
```

### Required fields

以下字段不能为空：

```text
schema_version
run_id
model_id
benchmark
phase
status
```

当 `status=completed` 时，以下字段必须填写：

```text
provenance.started_at
provenance.finished_at
counts.expected_episodes
counts.recorded_episodes
counts.successes
counts.failures
counts.errors
```

如果 checkpoint 信息无法公开，允许保留：

```json
{
  "name": "<private>",
  "revision": null,
  "sha256": null
}
```

禁止写入：

```text
绝对路径
服务器 IP
SSH 用户名
访问 token
API key
私有日志路径
```

## 6. `episodes.jsonl` Protocol

`episodes.jsonl` 使用 JSON Lines 格式。

每行必须是一个独立、合法的 JSON 对象。

### Required episode fields

```json
{
  "schema_version": 1,
  "run_id": "act-libero-full-v1",
  "model_id": "act",
  "benchmark": "libero",
  "phase": "full",

  "episode_id": "libero:libero_spatial:task-00:trial-00:seed-7",
  "suite": "libero_spatial",
  "task_id": 0,
  "trial": 0,
  "seed": 7,

  "status": "success",
  "success": true,

  "action_steps": 120,
  "duration_seconds": 8.2,

  "error_category": null,
  "error_message": null,

  "video": "videos/act/libero/act-libero-full-v1/episode-000000.mp4",
  "video_available": true,
  "video_status": "written"
}
```

### Allowed status values

```text
success
failure
error
cancelled
not_recorded
```

### Allowed video status values

```text
written
not_recorded
missing
corrupt
not_requested
```

### Episode invariants

必须满足：

```text
success=true  => status=success
success=false => status != success
video_available=true => video_status=written
video_available=false => video_status != written
```

如果 episode 没有视频：

```json
{
  "video": null,
  "video_available": false,
  "video_status": "not_recorded"
}
```

禁止创建虚假的占位视频。

### Optional fields

模型或 benchmark 特有字段必须放入 `extra`：

```json
{
  "extra": {
    "action_chunk_size": 10,
    "observation_latency_ms": 42,
    "controller_frequency_hz": 20
  }
}
```

不得随意修改公共字段含义。

## 7. `summary.json` Protocol

标准格式：

```json
{
  "schema_version": 1,
  "run_id": "act-libero-full-v1",
  "model_id": "act",
  "benchmark": "libero",
  "phase": "full",

  "total": 2000,
  "successes": 1800,
  "failures": 190,
  "errors": 10,
  "success_rate": 0.9,

  "videos_available": 1995,
  "videos_missing": 5,

  "by_suite": {
    "libero_spatial": {
      "total": 500,
      "successes": 470,
      "failures": 28,
      "errors": 2,
      "success_rate": 0.94
    }
  }
}
```

汇总值必须由 `episodes.jsonl` 重新计算得到，不能手工填写。

必须满足：

```text
total = successes + failures + errors
success_rate = successes / total
videos_available + videos_missing = total
```

## 8. `summary.csv` Protocol

CSV 第一行必须是：

```csv
suite,total,successes,failures,errors,success_rate,videos_available,videos_missing
```

示例：

```csv
libero_spatial,500,470,28,2,0.94,498,2
libero_object,500,460,35,5,0.92,499,1
libero_goal,500,480,20,0,0.96,500,0
libero_10,500,390,107,3,0.78,498,2
```

## 9. Registry Protocol

在：

```text
results/run-registry.json
```

维护所有新模型结果。

初始模板：

```json
{
  "schema_version": 1,
  "runs": [
    {
      "run_id": "act-libero-full-v1",
      "model_id": "act",
      "benchmark": "libero",
      "phase": "full",
      "status": "planned",
      "path": "results/runs/act/libero/act-libero-full-v1"
    },
    {
      "run_id": "act-libero-plus-full-v1",
      "model_id": "act",
      "benchmark": "libero-plus",
      "phase": "full",
      "status": "planned",
      "path": "results/runs/act/libero-plus/act-libero-plus-full-v1"
    },
    {
      "run_id": "act-libero-pro-full-v1",
      "model_id": "act",
      "benchmark": "libero-pro",
      "phase": "full",
      "status": "planned",
      "path": "results/runs/act/libero-pro/act-libero-pro-full-v1"
    },
    {
      "run_id": "openvla-libero-full-v1",
      "model_id": "openvla",
      "benchmark": "libero",
      "phase": "full",
      "status": "planned",
      "path": "results/runs/openvla/libero/openvla-libero-full-v1"
    },
    {
      "run_id": "openvla-libero-plus-full-v1",
      "model_id": "openvla",
      "benchmark": "libero-plus",
      "phase": "full",
      "status": "planned",
      "path": "results/runs/openvla/libero-plus/openvla-libero-plus-full-v1"
    },
    {
      "run_id": "openvla-libero-pro-full-v1",
      "model_id": "openvla",
      "benchmark": "libero-pro",
      "phase": "full",
      "status": "planned",
      "path": "results/runs/openvla/libero-pro/openvla-libero-pro-full-v1"
    }
  ]
}
```

Smoke 测试可以额外增加对应的 `smoke` 记录。

## 10. Run Lifecycle

所有 AI 必须按照以下顺序处理：

### Phase A: Initialize

1. 创建 run 目录。
2. 写入 `run.json`。
3. 设置：

```json
"status": "planned"
```

### Phase B: Start

开始实际运行前，将状态改为：

```json
"status": "running"
```

并填写：

```text
provenance.started_at
```

### Phase C: Episode Recording

每完成一个 episode：

1. 生成完整 JSON 对象。
2. 写入临时文件。
3. 原子追加或替换到 `episodes.jsonl`。
4. 视频写盘完成后才能设置 `video_available=true`。
5. 不能因为模型成功而假设视频一定存在。

### Phase D: Resume

断点续跑时：

1. 读取已有 `episodes.jsonl`。
2. 使用 `episode_id` 去重。
3. 已经存在且字段完整的 episode 不得重复写入。
4. 不得覆盖已有成功结果，除非明确使用新的 `run_id`。
5. 中途失败应保留当前结果，并设置：

```json
"status": "partial"
```

### Phase E: Finalize

完成后：

1. 重新读取全部 `episodes.jsonl`。
2. 重新计算 `summary.json`。
3. 重新生成 `summary.csv`。
4. 校验视频数量。
5. 更新 `run.json` 的 counts。
6. 设置：

```json
"status": "completed"
```

## 11. Validation Requirements

必须提供一个校验命令：

```bash
python3 scripts/validate_run.py \\
  --run-dir results/runs/act/libero/act-libero-full-v1
```

校验器至少检查：

```text
run.json 是否存在
episodes.jsonl 是否逐行合法 JSON
run_id 是否一致
model_id 是否一致
benchmark 是否一致
phase 是否一致
episode_id 是否重复
success/status 是否矛盾
视频字段是否一致
summary.json 是否与 episodes.jsonl 一致
summary.csv 是否存在
绝对路径是否泄漏
token/API key 是否泄漏
```

校验成功时输出：

```text
VALID
run_id=act-libero-full-v1
episodes=2000
successes=1800
failures=190
errors=10
videos_available=1995
```

任何校验失败都必须返回非零退出码。

## 12. Existing OpenPI Results

现有 pi0.5 结果保持不动：

```text
results/openpi-libero/
results/plus-pro/
results/libero-x/
```

不得为了适配新协议而移动、重命名或重写旧结果。

新协议只用于：

```text
results/runs/act/
results/runs/openvla/
```

后续报告程序可以通过 adapter 同时读取旧格式和本协议格式。

## 13. AI Implementation Instructions

执行本协议的 AI 必须遵守：

1. 不修改已有 pi0.5 结果。
2. 不把 checkpoint、模型权重或 benchmark 资产提交到 GitHub。
3. 不把视频提交到 GitHub。
4. 不提前假设 ACT/OpenVLA 的启动命令。
5. 不修改公共字段含义。
6. 不制造占位视频。
7. 不把失败 episode 删除。
8. 不把 N/A episode 当成失败。
9. 不覆盖已有 run，新的配置使用新的 `run_id`。
10. 运行结束后必须生成 `summary.json` 和 `summary.csv`。
11. 运行完成前不得标记为 `completed`。
12. 所有最终数字必须由 `episodes.jsonl` 重新统计得到。

## 14. Acceptance Criteria

接口实现完成后，至少应满足：

```text
能够创建 ACT/LIBERO 的 planned run
能够创建 OpenVLA/LIBERO-Plus 的 planned run
能够写入单条 episode
能够断点续跑且不产生重复 episode
能够处理没有视频的 episode
能够从 episodes.jsonl 自动生成 summary.json
能够校验 run.json 和 episodes.jsonl
不会影响现有 pi0.5 结果
不会把视频和 checkpoint 纳入 GitHub
```
