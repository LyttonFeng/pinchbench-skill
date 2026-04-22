# Task 18 Focused DPO Failure Summary

## Core Observation

We trained a task-specific focused DPO LoRA for `task_18_spreadsheet_summary` on top of `Qwen3-1.7B-Instruct`.

The DPO training clearly **converged numerically**, and the LoRA adapter was **correctly loaded into vLLM**, but under the real `OpenClaw + vLLM + tool parser` runtime, the model's first action still did **not** switch to the desired:

- `exec + pandas.read_excel(...)`

Instead, it still followed the base-policy behavior:

- `read quarterly_sales.csv`
- `read company_expenses.xlsx`

Final benchmark score on `task_18_spreadsheet_summary` remained:

- `2.5%`

## What We Already Verified

### 1. Training did converge

Bootstrap focused DPO dataset:

- `59` pairs total
- negative types:
  - `read_csv_first: 29`
  - `read_xlsx: 10`
  - `session_status: 10`
  - `explore_ls: 10`

Training result:

- `train_loss = 0.02674`
- reward margins became very large
- reward accuracy reached `1`

So this is **not** a case where training failed to run.

### 2. LoRA deployment worked

We confirmed:

- adapter files were complete
- LoRA was loaded by vLLM
- `/v1/models` showed the LoRA model correctly

So this is **not** a case where the adapter failed to load.

### 3. Real runtime evaluation still failed

Direct sanity with the real runtime-style request still returned:

- `read quarterly_sales.csv`
- `read company_expenses.xlsx`

And real benchmark result on `task_18_spreadsheet_summary` was:

- `0.025 / 1.0 = 2.5%`

Judge notes:

- model treated the Excel file like binary/text
- failed to compute required statistics
- failed to produce the required report

## Runtime / Training Setup

Model:

- `Qwen3-1.7B-Instruct`

Task:

- `task_18_spreadsheet_summary`

Inference stack:

- `OpenClaw + vLLM + tool parser`

Focused DPO data source:

- harness-driven collection from the real runtime path
- real runtime requests were captured from the actual `OpenClaw` path

Important note:

- real runtime request had `24` tools
- training used harness-captured `messages + tools`

## Remaining Mismatch / Risk

The main known residual mismatch is:

- training used prompt truncation
- inference used the full runtime prompt

In training, prompt was truncated to preserve the completion region.
So this is still **not perfectly full train-infer parity**.

## Most Likely Explanations

### 1. Numerically converged DPO did not flip the discrete tool-choice boundary

The model learned the pairwise preference objective, but that preference did not become a real first-action switch under the production runtime.

### 2. Residual train-infer mismatch still matters

Even though we moved much closer to runtime parity by using harness-captured `messages + tools`, prompt truncation may still mean the model is not being trained on the exact same effective condition distribution as inference.

### 3. The base model prior is too strong

For this task, `Qwen3-1.7B-Instruct` appears to have a strong prior toward:

- read CSV first
- read XLSX directly as a file

That prior may still dominate at decode time, even after DPO convergence.

### 4. Focused DPO on first action may still be insufficient

Training optimized only the first assistant action preference, while benchmark success depends on the whole task.

If the first-step switch is not fully achieved, the whole task still collapses.

### 5. Negative distribution may still be imperfect

Although the bootstrap set is multitype, the real harness data was still dominated by:

- `read_csv_first`

So the model may have learned a partial preference pattern, but not a strong enough “always use exec for xlsx workbook parsing” policy.

## Concrete Question

Given the following facts:

- DPO training converges numerically
- LoRA is correctly loaded
- real runtime direct sanity still follows base-policy tool choices
- benchmark score remains `2.5%`
- training data uses real harness-captured runtime prompts/tools, but prompt truncation still exists

What is the most likely explanation for this failure mode?

In particular:

1. Why would DPO convergence fail to translate into a real tool-choice change at inference time?
2. Is prompt truncation likely the main blocker, or is this more likely a “base prior too strong / objective too weak” issue?
3. For this kind of agentic tool-selection problem, would you expect:
   - more real harness data
   - stronger negative coverage
   - SFT + DPO combination
   - ORPO / GRPO / RL
   to be more effective than focused DPO alone?
