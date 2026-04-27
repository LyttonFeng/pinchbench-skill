# Fork Repo Cleanup Plan

日期：2026-04-27

目标：降低 fork repo 对协作者的认知干扰。远端只保留可复现主线、算法实现、正式实验结论；阶段性讨论、handoff、demo 素材、单次实验碎片和生成物保留在本地，不再进入远端。

## 1. 保留在远端

这些文件构成当前 RL 工作的主线，应该保留：

```text
docs/rl_algo_readme.md
docs/task16_targeted_data_plan_v3_20260423.md
docs/task16_task18_baseline_vs_skilllike_20260423.md

rl/train/run_reinforce_task16_event_only_v2.sh
rl/train/run_reinforce_lora.sh
rl/train/build_task16_variant_prompts.py
rl/train/launch_main_ppo.py
rl/train/reward_manager.py

rl/agent_loop/openclaw_agent_loop.py
rl/agent_loop/reward.py
rl/agent_loop/task16_event_reward/reward_task16_event_only_v2.py
rl/verl_no_masked_whiten_patch.py

scripts/benchmark.py
scripts/run_bench_rl8.sh
scripts/run_bench_rl8_lora.sh
scripts/lib_agent.py
```

保留理由：

- 能解释算法设计
- 能复现实验
- 是训练 / rollout / reward / benchmark 主链路
- 对协作者理解当前有效路径有帮助

## 2. 从远端移除，但本地归档保留

这些文件更像阶段性沟通记录、临时 handoff、demo 素材、旧实验碎片。建议用 `git rm --cached` 从远端移除，同时复制到本地 `local_archive/`。

### 2.1 Handoff / 临时排障记录

```text
docs/CODEX_RUNPOD_HANDOFF.md
docs/rl_openclaw_handover.md
docs/task18_harness_connectivity_20260422.md
```

理由：对当前主线复现帮助有限，且包含大量时效性环境信息，容易误导后续协作者。

### 2.2 Demo / 展示素材

```text
docs/task16_demo_slides.html
docs/task16_email_triage_demo_plan_20260422.md
```

其中远端只保留：

```text
docs/task16_demo_slides.html
```

移出远端、本地保留：

```text
docs/task_16_email_triage_demo.html
docs/task16_email_triage_demo_plan_20260422.md
```

理由：`task16_demo_slides.html` 最接近最终展示物；另外两个分别是早期 demo 和计划文档，保留在远端会和最终版本冲突。

### 2.3 旧实验报告 / 阶段性分析

```text
docs/experiment_summary_v1_v2_20260416.md
docs/qwen3_1_7b_rl8_think_vs_nonthink_20260419.md
docs/qwen3_1_7b_rl_milestone_20260420.md
docs/rl8_reinforcepp_20260413_experiment_report.md
docs/task18_dpo_experiment_20260420.md
docs/task18_focused_dpo_data_design_20260422.md
docs/task18_focused_dpo_train_infer_parity_20260422.md
docs/task18_dpo_failure_summary_for_external_review_20260422.md
docs/task18_reward_event_table_20260422.md
docs/task16_reward_trim_plan_20260423.md
```

理由：这些结论已经被后续 `rl_algo_readme.md` 和 task16 主线文档吸收。远端保留太多会让协作者无法判断哪一版是当前结论。

### 2.4 草稿 / 公众号 / 开放性讨论

```text
docs/jiuwen_claw_vs_hermes_discussion.md
docs/jiuwenclaw_rl_wechat_draft.md
docs/jiuwenclaw_self_evolution_framework.md
docs/self_evolution_data_generation_plan.md
docs/spreadsheet_data_production_module.md
```

理由：适合个人思考和报告，不适合和训练主链路混在一起。

### 2.5 单次 baseline 结果

```text
results/BASELINE_SUMMARY_Qwen3-4B_20260408.md
```

理由：结果目录应默认不进 repo。正式对比结论应汇总进 docs，而不是提交单次 result 文件。

## 3. 当前未提交，但建议加入 `.gitignore`

这些当前多为本地生成物或探索项目，不应误提交。

```text
.openclaw/
.pinchbench_runs/
scripts/results/
local_archive/

assets/rl_ss_task_18_spreadsheet_summary-train-*
tasks/task_18_spreadsheet_summary-train-*

wolfpack/
```

说明：

- `.openclaw/`、`.pinchbench_runs/` 是运行时状态。
- `scripts/results/` 是 benchmark 中间结果。
- `assets/rl_ss_task_18_spreadsheet_summary-train-*` 和 `tasks/task_18_spreadsheet_summary-train-*` 是生成数据，不应混入 canonical tasks/assets。
- `wolfpack/` 当前还处于探索期；等抽象稳定后再决定是否单独提交。
- `local_archive/` 用来保存从远端移除但本地仍需参考的材料。

## 4. 建议执行步骤

### Step 1: 建本地归档目录

```bash
mkdir -p local_archive/docs local_archive/results
```

### Step 2: 复制待移除文件到本地归档

```bash
cp docs/CODEX_RUNPOD_HANDOFF.md local_archive/docs/
cp docs/rl_openclaw_handover.md local_archive/docs/
cp docs/task18_harness_connectivity_20260422.md local_archive/docs/
```

其余待移除 docs 同理复制。

### Step 3: 从 git index 移除，但保留本地文件

```bash
git rm --cached <path>
```

不要用 `rm`，避免本地分析材料丢失。

### Step 4: 更新 `.gitignore`

追加：

```gitignore
local_archive/
.openclaw/
.pinchbench_runs/
scripts/results/
assets/rl_ss_task_18_spreadsheet_summary-train-*
tasks/task_18_spreadsheet_summary-train-*
wolfpack/
```

### Step 5: 单独提交 cleanup commit

```bash
git add .gitignore docs/repo_cleanup_plan_20260427.md
git commit -m "docs: add fork cleanup plan"
```

如果确认执行清理，再提交：

```bash
git commit -m "chore: remove local experiment artifacts from fork"
git push fork main
```

## 5. 当前不建议做的事

```text
git reset --hard
git clean -fd
git rm <path>
git add .
```

原因：

- 当前工作树里有大量未提交实验状态
- 有些材料本地还要继续分析
- `git add .` 会把生成数据和探索代码误推上去

## 6. 推荐远端文档结构

清理后，远端 docs 建议只保留三类：

```text
docs/rl_algo_readme.md
  算法 + 复现实验 runbook

docs/task16_targeted_data_plan_v3_20260423.md
  task16 数据构造和 failure taxonomy

docs/task16_task18_baseline_vs_skilllike_20260423.md
  正式实验结果表和主结论

docs/task16_demo_slides.html
  demo 展示页
```

其他报告、demo、handoff、pod debug、公众号草稿，统一放本地 `local_archive/`。
