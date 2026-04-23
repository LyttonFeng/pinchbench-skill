# task_16 / task_18 Baseline vs Skill-Like Prompt

Date: 2026-04-23  
Model: `Qwen/Qwen3-1.7B`  
Inference path: `Mac -> 127.0.0.1:18025 -> L40S vLLM(8021)`  
Service model: baseline `Qwen3-1.7B` only, no LoRA

## Scope

This note compares:

1. baseline prompt
2. skill-like prompt

for:

1. `task_16_email_triage`
2. `task_18_spreadsheet_summary` (disk markdown file: `task_19_spreadsheet_summary.md`)

All benchmark runs were executed from the Mac side. This avoids the remote-benchmark path drift that showed up earlier.

## task_16_email_triage

### Extracted skill-like prompt

Injected procedure:

1. read the inbox efficiently and avoid repeatedly rereading the full inbox
2. identify highest-signal items first:
   - production incident / outage
   - correlated monitoring alert
   - high-value client thread
   - security / compliance deadline
   - release-blocking code review
3. explicitly link emails that belong to the same operational incident
4. make business impact explicit in prioritization
5. task is not complete until `triage_report.md` exists and covers the full inbox

### Baseline x3

| Run | Result file | Score |
| --- | --- | --- |
| 1 | `results/0184_qwen-qwen3-1-7b.json` | `60.0%` |
| 2 | `results/0185_qwen-qwen3-1-7b.json` | `52.3%` |
| 3 | `results/0186_qwen-qwen3-1-7b.json` | `36.0%` |

Mean: `49.4%`

### Skill-like x3

| Run | Result file | Score |
| --- | --- | --- |
| 1 | `results/0193_qwen3-1-7b.json` | `36.0%` |
| 2 | `results/0194_qwen3-1-7b.json` | `36.0%` |
| 3 | `results/0195_qwen3-1-7b.json` | `36.0%` |

Mean: `36.0%`

### Interpretation

The skill-like prompt did **not** improve `task_16`.

Observed failure mode remained stable:

1. the model repeatedly read the inbox
2. it often demonstrated partial prioritization ability
3. but it still failed to reliably produce `triage_report.md`

This means the bottleneck is not just "what to pay attention to". The bigger issue is execution closure: converting inbox reading into a finished artifact.

### Revised skill-like prompt (v2)

The first skill-like prompt was too high-level. A second prompt revision made the operating constraints more explicit:

1. do one fast inbox pass first
2. avoid full inbox rereads unless a specific email must be rechecked
3. build a working list that covers all 13 emails
4. explicitly identify outage, correlated alert, high-value client, security deadline, and release blocker
5. once coverage is sufficient, stop reading and write `triage_report.md`
6. require every email to have priority, category, and recommended action
7. make high business impact outrank routine noise

### Revised skill-like x3 (v2)

| Run | Result file | Score |
| --- | --- | --- |
| 1 | `results/0203_qwen3-1-7b.json` | `36.0%` |
| 2 | `results/0204_qwen3-1-7b.json` | `60.1%` |
| 3 | `results/0205_qwen3-1-7b.json` | `49.3%` |

Mean: `48.5%`

### Revised interpretation

The revised `task_16` skill-like prompt is materially better than the first version.

Compared with the first skill-like prompt:

1. mean improved from `36.0%` to `48.5%`
2. one run matched/exceeded the baseline high end (`60.1%`)
3. forcing an explicit "stop reading, start writing" transition appears more useful than only emphasizing high-signal items

However, it still does not reliably beat baseline:

1. one run remained stuck at `36.0%`
2. failure notes still show missed linkage and omitted high-value emails
3. execution remains unstable

So for `task_16`, the better prompt direction is:

- less abstract prioritization advice
- more explicit workflow control:
  - cover inbox once
  - stop rereading
  - produce a complete artifact

### RL best checkpoint x3 (`task16-rl-step24`)

Best RL run:

1. experiment: `reinforce_lora_task16_event_only_v2_qwen31`
2. best checkpoint: `global_step_24`

The first RL benchmark run (`results/0206_task16-rl-step24.json`, `41.9%`) is excluded here. It was the first run immediately after service hot-swap and looked noisier than the later runs. To keep the comparison fair, RL is also summarized with 3 runs.

| Run | Result file | Score |
| --- | --- | --- |
| 1 | `results/0207_task16-rl-step24.json` | `43.8%` |
| 2 | `results/0208_task16-rl-step24.json` | `57.1%` |
| 3 | `results/0209_task16-rl-step24.json` | `56.8%` |

Mean: `52.6%`

### RL interpretation

This is the first `task_16` result in this project line that is modestly above the baseline mean.

Comparison:

1. baseline x3 mean: `49.4%`
2. revised skill-like x3 mean: `48.5%`
3. RL best-ckpt x3 mean: `52.6%`

Trajectory-level errors that still remain:

1. `email_13` correlated latency alert is still sometimes omitted or under-prioritized
2. `email_01` outage and `email_13` alert are not stably linked into one P0 incident
3. `email_05` BigClient can still be downgraded or missed
4. `email_08` security/compliance deadline is still not reliably escalated high enough

What did improve:

1. report completion is more stable
2. repeated full-inbox reread behavior appears reduced
3. the model is more likely to produce a usable `triage_report.md`

### RL best checkpoint + v2 hint x3 (`task16-rl-step24`)

This comparison keeps the RL checkpoint fixed at `global_step_24` and injects the revised v2 task hint into the task prompt at benchmark time.

| Run | Result file | Score |
| --- | --- | --- |
| 1 | `results/0210_task16-rl-step24.json` | `51.5%` |
| 2 | `results/0211_task16-rl-step24.json` | `60.0%` |
| 3 | `results/0212_task16-rl-step24.json` | `42.1%` |

Mean: `51.2%`

### RL + hint interpretation

Adding the revised v2 hint on top of the RL checkpoint did **not** create another step-change.

Comparison:

1. baseline x3 mean: `49.4%`
2. revised skill-like x3 mean: `48.5%`
3. RL best-ckpt x3 mean: `52.6%`
4. RL best-ckpt + v2 hint x3 mean: `51.2%`

Interpretation:

1. the RL checkpoint already learned part of the workflow control signal
2. adding the hint does not reliably improve it further
3. one run (`0211`) was very strong, but another (`0212`) drifted badly, so the hint is not acting like a stable booster

Failure notes from the weak run remain familiar:

1. category drift
2. wrong priority assignment on critical items
3. unstable linkage behavior

## task_18_spreadsheet_summary

### Extracted skill-like prompt

Injected procedure:

1. treat `quarterly_sales.csv` and `company_expenses.xlsx` differently
2. `read` is acceptable for the CSV
3. do **not** use `read` on `.xlsx`
4. use `exec` with Python plus `pandas` / `openpyxl` for workbook inspection
5. compute required aggregates before writing
6. task is not complete until `data_summary.md` exists and contains computed results

### Baseline x3

| Run | Result file | Score |
| --- | --- | --- |
| 1 | `results/0197_qwen3-1-7b.json` | `2.5%` |
| 2 | `results/0198_qwen3-1-7b.json` | `2.5%` |
| 3 | `results/0199_qwen3-1-7b.json` | `2.5%` |

Mean: `2.5%`

### Skill-like x3

| Run | Result file | Score |
| --- | --- | --- |
| 1 | `results/0200_qwen3-1-7b.json` | `2.5%` |
| 2 | `results/0201_qwen3-1-7b.json` | `12.5%` |
| 3 | `results/0202_qwen3-1-7b.json` | `2.5%` |

Mean: `5.8%`

### Interpretation

The skill-like prompt produced a **small but real directional improvement** on `task_18`, but it was unstable.

What changed:

1. one run improved from the baseline band of `2.5%` to `12.5%`
2. judge notes show the model moved slightly closer to the desired route in that run

What did not change:

1. the model still frequently used `read` on the `.xlsx`
2. it still failed to robustly parse the workbook
3. it still often failed to generate a valid `data_summary.md`

So for `task_18`, skill-like guidance helps routing pressure, but it does not fix execution quality.

## Bottom line

### task_16

`task_16` now has three relevant reference points:

1. baseline x3 mean: `49.4%`
2. revised skill-like x3 mean: `48.5%`
3. RL best-ckpt x3 mean: `52.6%`
4. RL best-ckpt + v2 hint x3 mean: `51.2%`

So the current conclusion is:

1. old skill-like prompting was bad
2. revised skill-like prompting recovered to roughly baseline level
3. RL v2 produced a small but real lift above baseline, but not a large one
4. adding the v2 hint on top of the RL checkpoint did not improve the mean further

The current bottleneck is no longer just "write the report at all". It is:

1. stable incident linkage
2. stable handling of `email_13`
3. stable prioritization of BigClient and security items

### task_18

Skill-like prompting is more promising than on `task_16`, because it can sometimes move the model in the right direction.  
But the effect is weak and noisy. It is not enough to make the task reliable.

## Notes

1. `results/0196_qwen3-1-7b.json` is excluded. It was a wrong entrypoint run (`0/0`) caused by using the RL8 wrapper for a non-RL8 single-task benchmark.
2. Temporary skill-like prompt edits to `tasks/task_16_email_triage.md` and `tasks/task_19_spreadsheet_summary.md` were restored after benchmarking.
3. `task_16` was benchmarked twice with different skill-like prompt versions; the revised v2 prompt is the more relevant one going forward.
4. `results/0206_task16-rl-step24.json` is excluded from the RL summary to keep the RL comparison on the same 3-run footing as baseline and skill-like.
5. `results/0210_task16-rl-step24.json` through `results/0212_task16-rl-step24.json` are the RL-checkpoint-plus-v2-hint runs.
