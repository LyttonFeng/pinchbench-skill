# v3 Rubric Fix — Handoff

## Status: DONE — rubric edits applied, ready to commit and train

---

## What was done

All 4 rubric fixes applied to `rl/agent_loop/reward.py` → `TASK_RUBRICS`:

### task_02_stock
- `optional_hints`: added sentence — "writing an approximate report always beats writing nothing"
- `common_mistakes`: prepended refusal-is-worst-outcome entry (model was refusing to write file citing "privacy constraints")

### task_12_skill_search
- `common_mistakes`: prepended 2 new entries:
  1. Don't call `read('config/*')` — use exact filenames from the task prompt
  2. Don't give up after ENOENT — files are listed in the prompt, use them directly

### task_18_spreadsheet_summary
- `optional_hints`: replaced entirely — removed impossible references to "real parser/library" and "awk" (neither exists in OpenClaw); now says: read CSV for real numbers, xlsx returns binary (expected), use prompt-provided structure for xlsx estimates, partial report beats nothing
- `common_mistakes`: removed 2 impossible entries ("Not retrying with another tool when first exec fails", "Filling gaps with guessed numbers after a parsing error instead of re-parsing"); added 3 real ones targeting the infinite thinking loop and binary-xlsx failure mode

### task_24_polymarket_briefing
- `common_mistakes`: prepended 3 new entries targeting date hallucination (wrong year in header), 2023-era stale knowledge fallback, and stopping after only 2 search attempts

---

## Root causes (from v2 trajectory analysis)

| Task | v1→v2 score | Root cause |
|------|-------------|------------|
| task_02_stock | 92%→0% | Model learned refusal is safer than wrong price; rubric didn't penalize refusal explicitly |
| task_12_skill_search | 17%→0% | Used glob path `config/*` → ENOENT → gave up; rubric didn't cover file-discovery failure |
| task_18_spreadsheet_summary | 2.5%→0% | Rubric referenced impossible tools (awk, exec retry); xlsx binary → infinite thinking loop |
| task_24_polymarket_briefing | 54%→21% | Date hallucination (wrote 2023-10-25); fell back to 2023 training data when search was poor |

---

## Next steps

1. `git commit` the reward.py changes
2. Kill v2 RunPod job (already done / check)
3. Launch v3 training — same hyperparams as v2: `BATCH=4, KL, turn-level, oracle-judge`
4. Monitor val-core at step 2/4/6/8
5. Benchmark best checkpoint with RL8 suite

---

## Committee judging (deferred to v4 if needed)

If v3 scores are still unstable after rubric fix, add `n_votes=2` ensemble to `compute_prm_reward_oracle_judge()` in `reward.py` — call oracle judge twice per turn (temperature>0), average the scores. Cost doubles but variance drops.

---

## OpenClaw Catalog Model Registration — Common Failure Modes

Benchmarking a vLLM-served LoRA requires three things to align: the SSH tunnel port, the OpenClaw agent config, and the vLLM `--lora-modules` alias. Any mismatch causes silent failures ("Transcript not found" or 0-score runs).

### How it works

```
Mac benchmark script
  → OpenClaw ECS agent (bench-pinchbench-lora)
    → model id: "pinchbench-lora"
    → baseUrl: http://127.0.0.1:18021/v1
      → SSH tunnel: 127.0.0.1:18021 → RunPod:8021
        → vLLM serving LoRA adapter
```

### Agent config location (on ECS 8.163.82.224)

```
~/.openclaw/agents/bench-pinchbench-lora/agent/models.json
```

Key fields:
```json
{
  "baseUrl": "http://127.0.0.1:18021/v1",
  "model": "pinchbench-lora"
}
```

### vLLM startup (on RunPod)

vLLM must register the LoRA adapter with the **exact alias** that the agent expects:

```bash
--lora-modules pinchbench-lora=/path/to/lora_adapter
```

If you use a different alias (e.g. `Qwen3-4B=...`), vLLM serves it under `Qwen3-4B` and the agent's `pinchbench-lora` model id returns 404 → silent 0-token failure.

### SSH tunnel (on Mac)

```bash
ssh -N -L 18021:localhost:8021 root@216.81.248.115 -p 15416 -i ~/.ssh/id_ed25519
```

Port mapping: **local 18021 → RunPod vLLM 8021**. Do NOT use 18010 (that's the base model tunnel).

### Running the benchmark

```bash
MODEL=pinchbench-lora BASE_URL=http://127.0.0.1:18021/v1 bash scripts/run_bench_rl8.sh
```

### Checklist before each benchmark run

1. vLLM running on RunPod port 8021 with `--lora-modules pinchbench-lora=<adapter_path>`
2. SSH tunnel active: `127.0.0.1:18021 → RunPod:8021`
3. Verify model is reachable: `curl http://127.0.0.1:18021/v1/models` → should list `pinchbench-lora`
4. Run benchmark with `MODEL=pinchbench-lora BASE_URL=http://127.0.0.1:18021/v1`
