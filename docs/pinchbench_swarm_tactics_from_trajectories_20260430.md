# PinchBench Swarm Tactics From Trajectories (2026-04-30)

## TL;DR

这份设计只基于任务 prompt、workspace 可见文件、agent 轨迹和工具错误，不基于 grading 细则做答案 hack。

目标不是让 controller 直接解题，而是让 controller 把 1.7B 小模型的失败模式工程化拆开：

1. 先做 workspace discovery。
2. 再由多个 1.7B worker 产出结构化中间证据。
3. controller 只做合并、冲突检测、闭环检查和产物写入约束。
4. 最后由 writer worker 生成用户可见 artifact。
5. critic worker 只根据任务 prompt 和 shared state 检查“是否完成用户要求”，不看 grading rubric。

这适合验证一个核心问题：

> 不训练模型，只通过 evidence-grounded swarm workflow，能不能把单体 1.7B 的 tool-use / closure / long-context failure 降下来？

## Non-Hack Contract

允许：

- 读取任务 prompt。
- 读取 workspace 文件列表和文件内容。
- 读取历史轨迹中的 tool calls / tool results / errors。
- 总结失败模式，例如 path error、missing output file、binary file read、no final write、hallucinated web source。
- 根据任务类型制定通用工作流，例如 extract -> compute -> write -> verify。

不允许：

- 读取或硬编码 grading criteria。
- 硬编码 hidden expected answer。
- 让 controller 直接写最终答案里的关键事实，除非该事实来自 workspace evidence 或 worker outputs。
- 针对某个 fixture 写死具体数值、具体 email linkage、具体 anomaly badge。

## Shared State Schema

每个任务都维护一个 `state.json`，controller 是唯一写入者。

```json
{
  "task_id": "task_16_email_triage",
  "prompt_summary": "...",
  "workspace_manifest": [
    {"path": "inbox/email_01.md", "kind": "text", "read_status": "ok"}
  ],
  "evidence": [],
  "candidate_outputs": [],
  "conflicts": [],
  "final_artifacts": [
    {"path": "triage_report.md", "status": "written"}
  ],
  "closure_check": {
    "all_required_outputs_present": true,
    "unresolved_errors": [],
    "critic_notes": []
  }
}
```

## Generic Swarm Roles

| Role | 1.7B Worker Duty | Controller Duty |
|---|---|---|
| `discoverer` | List visible files and infer required outputs from prompt. | Build workspace manifest and prevent missing-file hallucination. |
| `reader/extractor` | Extract facts from assigned files. | Merge evidence, dedupe, mark unread files. |
| `planner` | Propose execution plan from prompt and evidence. | Choose safe plan and enforce step order. |
| `calculator/executor` | Compute numbers or transform structured data. | Validate arithmetic format and compare independent runs. |
| `linker` | Build cross-document links or dependency graph. | Merge graph and surface conflicts. |
| `writer` | Draft final artifact. | Ensure artifact path exists and required sections are present. |
| `critic` | Check final artifact against prompt and evidence. | Trigger targeted retry if output missing or inconsistent. |

## Stop Policy

Stop when all are true:

- Required output files exist.
- Every required input file is either read or explicitly marked unread with reason.
- No unresolved tool error blocks the final artifact.
- Critic says the final artifact answers the prompt.
- If a retry lowers evidence coverage or deletes required artifact, keep best-so-far state.

## Task-Specific Tactics

### task_00_sanity

Failure pattern: simple closure task; single agent already stable.

Swarm tactic:

- `discoverer`: identify required response or file.
- `writer`: perform minimal final action.
- `critic`: verify output exists or final answer directly satisfies prompt.

Shared state focus:

- `required_artifact`
- `completed_action`

### task_01_calendar

Failure pattern: event details can be dropped or malformed.

Swarm tactic:

- `extractor_a`: extract date/time/title/attendees/location.
- `extractor_b`: independently extract same fields.
- `calendar_writer`: create event artifact or action payload.
- `critic`: compare required event fields against prompt-derived field list.

Shared state focus:

- canonical event object
- missing fields
- final calendar action payload

### task_02_stock

Failure pattern: web/search/tool-use uncertainty and stale or unsupported quotes.

Swarm tactic:

- `query_planner`: derive target ticker/company and date requirement.
- `source_worker_1/2`: fetch or reason from available tool outputs independently.
- `reconciler`: compare ticker, price, date, source provenance.
- `writer`: produce concise answer with source timestamp if available.

Shared state focus:

- ticker/company mapping
- quote value
- quote timestamp
- source provenance

### task_03_blog

Failure pattern: output can be generic, overlong, or miss requested structure.

Swarm tactic:

- `outline_worker`: create outline from prompt.
- `detail_worker`: draft section facts and examples.
- `style_worker`: adapt tone and audience.
- `writer`: produce final blog.
- `critic`: check prompt coverage, section flow, and no placeholder text.

Shared state focus:

- outline
- required topics
- tone constraints
- final draft path or response

### task_04_weather

Failure pattern: model calls nonexistent tools or writes incomplete script.

Swarm tactic:

- `script_planner`: produce minimal Python design using `urllib` or `requests`.
- `code_writer`: write `weather.py`.
- `static_checker`: inspect script text for imports, URL, print, error handling.
- `repair_worker`: patch only missing static pieces.

Shared state focus:

- target file `weather.py`
- API URL
- script syntax status
- missing implementation pieces

No grading hack: checker validates generic script completeness from task prompt, not hidden expected score.

### task_05_summary

Failure pattern: incomplete reading or generic summary.

Swarm tactic:

- `discoverer`: enumerate source documents.
- `extractor_per_doc`: summarize each document into bullets.
- `synthesizer`: merge common themes and unique facts.
- `writer`: write final summary.
- `critic`: verify every source document has evidence in state.

Shared state focus:

- per-document evidence cards
- coverage matrix
- final summary

### task_06_events

Failure pattern: research task can hallucinate event details or miss constraints.

Swarm tactic:

- `constraint_extractor`: extract event topic, date range, geography, output format.
- `search_worker_1/2`: independently collect candidate events.
- `source_reconciler`: keep only candidates with evidence.
- `writer`: produce event list with dates, locations, links if available.

Shared state focus:

- query constraints
- candidate events
- evidence/source per event

### task_07_email

Failure pattern: tone or required points missing.

Swarm tactic:

- `intent_extractor`: extract recipient, goal, tone, must-include points.
- `draft_worker_1/2`: draft alternatives.
- `tone_editor`: improve professionalism and concision.
- `critic`: check must-include coverage and no unsupported commitments.

Shared state focus:

- email intent object
- must-include checklist
- final email draft

### task_08_memory

Failure pattern: facts are present in context but retrieval is brittle.

Swarm tactic:

- `fact_extractor`: extract all user facts from prompt/context.
- `memory_indexer`: store facts as key-value entries.
- `answer_worker`: answer requested item from index.
- `critic`: verify answer is grounded in memory index.

Shared state focus:

- fact table
- query target
- cited memory entry

### task_09_files

Failure pattern: directory/file closure errors.

Swarm tactic:

- `structure_planner`: derive directory tree from prompt.
- `file_writer`: create files.
- `manifest_checker`: list workspace and compare against planned tree.
- `repair_worker`: create missing files only.

Shared state focus:

- planned tree
- actual tree
- missing/extra files

### task_10_workflow

Failure pattern: multi-step API/file workflow loses state or stops early.

Swarm tactic:

- `workflow_planner`: decompose into ordered steps.
- `config_reader`: extract endpoint, auth, parameters from files.
- `script_worker`: produce executable script or notes.
- `runner/validator`: if tools allow, run or statically inspect output.
- `writer`: document process and final artifact.

Shared state focus:

- ordered workflow graph
- config facts
- output files
- unresolved tool errors

### task_11_clawdhub

Failure pattern: project scaffold may miss files or wrong names.

Swarm tactic:

- `scaffold_planner`: infer target project tree.
- `file_writer`: create directories/files.
- `manifest_checker`: compare actual tree to plan.
- `critic`: check files are non-empty and consistent.

Shared state focus:

- planned project tree
- actual project tree
- file content summaries

### task_12_skill_search

Failure pattern: search/replace can miss files or corrupt unrelated text.

Swarm tactic:

- `search_worker`: list matches with file/line/context.
- `replacement_planner`: propose replacements.
- `editor`: apply replacements.
- `diff_checker`: verify old pattern gone and unrelated files unchanged.

Shared state focus:

- match list
- edit plan
- post-edit verification

### task_13_image_gen

Failure pattern: model lacks actual image tool path or produces text-only response.

Swarm tactic:

- `asset_planner`: extract visual prompt, size/style/output requirement.
- `tool_router`: decide whether image generation tool exists in runtime.
- `fallback_writer`: if no image tool, write a clear failure note only if allowed by task.
- `critic`: verify actual asset file exists when required.

Shared state focus:

- required image artifact
- available tools
- generated asset path

This task is mostly runtime-tool dependent; swarm can only help route and close.

### task_14_humanizer

Failure pattern: does not install/use skill, writes placeholder, or fails to read original text.

Swarm tactic:

- `discoverer`: locate source article and available skills.
- `content_extractor`: read full original content.
- `rewrite_worker_1/2`: produce humanized drafts.
- `editor`: merge best phrasing while preserving facts.
- `critic`: compare original vs rewritten coverage and detect placeholder text.

Shared state focus:

- original content digest
- rewrite draft versions
- preservation checklist
- final output path

### task_15_daily_summary

Failure pattern: path errors, missing files, generic template.

Swarm tactic:

- `discoverer`: enumerate research files before reading.
- `extractor_per_file`: summarize each source with key facts.
- `priority_synthesizer`: rank themes by urgency/impact.
- `writer`: create daily summary.
- `critic`: verify every discovered source has representation or explicit exclusion.

Shared state focus:

- source manifest
- per-source cards
- theme ranking
- final briefing

### task_16_email_triage

Failure pattern: weak inbox-level world model, incident linkage unstable, priority propagation unstable, no-report collapse.

Swarm tactic:

- `email_extractors`: split inbox into chunks and extract per-email facts.
- `incident_linkers`: propose cross-email groups from evidence.
- `prioritizers`: independently assign priority/category/action.
- `reconciler`: merge disagreements into an incident map.
- `writer`: write triage report from incident map.
- `critic`: check all emails covered and final report exists.

Shared state focus:

- per-email evidence cards
- incident map
- priority votes
- final triage report

No hard-coded fixture answer: linkage and priority must be derived from email evidence.

### task_17_email_search

Failure pattern: wrong directory strategy, uses memory search instead of reading email files, misses relevant emails.

Swarm tactic:

- `discoverer`: list email directory/files.
- `query_parser`: extract search intent from prompt.
- `search_workers`: scan chunks of emails for relevance.
- `summary_worker`: summarize only selected evidence.
- `critic`: verify selected emails cite matching evidence.

Shared state focus:

- email manifest
- query terms / intent
- relevant email candidates
- evidence-backed summary

### task_18_market_research

Failure pattern: hallucinated sources, generic competitors, weak timestamp/source grounding.

Swarm tactic:

- `research_planner`: define market, competitors, dimensions.
- `source_workers`: gather candidate facts from available research/web outputs.
- `source_auditor`: reject unsupported claims.
- `strategy_synthesizer`: convert facts into comparisons and implications.
- `writer`: write market research brief with citations/provenance.

Shared state focus:

- competitor profiles
- source provenance
- unsupported claims
- final brief

### task_18_spreadsheet_summary

Failure pattern: `.xlsx` read as binary garbage; model does not route to structured parser.

Swarm tactic:

- `file_router`: classify file types before reading.
- `csv_worker`: parse CSV text.
- `xlsx_worker`: request structured extraction path; if unavailable, mark runtime limitation.
- `calculator`: compute aggregates from structured tables.
- `writer`: write data summary.
- `critic`: verify all required input files were parsed structurally, not as raw binary.

Shared state focus:

- file type manifest
- table extraction status
- aggregate calculations
- runtime limitations

This task needs runtime support for `.xlsx`; swarm should avoid dumping binary into model context.

### task_20_eli5_pdf_summary

Failure pattern: PDF/tool output not converted into usable text; no final file.

Swarm tactic:

- `file_router`: identify PDF and text extraction method.
- `extractor`: obtain text chunks or declare extraction failure.
- `simplifier`: rewrite each chunk into child-friendly explanation.
- `writer`: assemble `eli5_summary.txt`.
- `critic`: verify file exists and uses simple language.

Shared state focus:

- PDF extraction status
- chunk summaries
- final ELI5 artifact

### task_21_openclaw_comprehension

Failure pattern: does not parse report/document, output missing or malformed.

Swarm tactic:

- `document_reader`: extract sections and key claims.
- `question_parser`: extract required answer fields.
- `answer_worker`: answer from document evidence.
- `critic`: verify every answer cites a document section.

Shared state focus:

- document section map
- question list
- evidence-backed answers

### task_22_second_brain

Failure pattern: memory write/read can overwrite, truncate, or fail across sessions.

Swarm tactic:

- `memory_writer`: write canonical memory file with all facts.
- `memory_checker`: reread file and verify all facts persisted.
- `session_answerer`: answer later questions from memory state.
- `critic`: check no fact was lost across sessions.

Shared state focus:

- canonical memory facts
- persisted file content hash/summary
- session answers

### task_24_polymarket_briefing

Failure pattern: hallucinated markets/news links and stale/no odds.

Swarm tactic:

- `query_planner`: define required market/news scope.
- `market_source_worker`: collect candidate markets only from available search/tool evidence.
- `news_source_worker`: collect news candidates with timestamp/source.
- `source_auditor`: reject unsupported market URLs, missing odds, or generic homepage links.
- `brief_writer`: write concise briefing.

Shared state focus:

- market candidates with odds/source/time
- news candidates with source/time
- rejected hallucinations
- final briefing

### task_25_access_log_anomaly

Failure pattern: CSV is readable, but model misuses path or stops without writing JSON.

Swarm tactic:

- `csv_reader`: read `access_events.csv` into rows.
- `rule_parser`: extract anomaly definitions from prompt.
- `detector_workers`: independently scan rows for each anomaly family.
- `json_writer`: write `anomaly_report.json`.
- `critic`: validate JSON parses and every finding cites row evidence.

Shared state focus:

- parsed rows
- rule definitions
- candidate anomalies
- final JSON artifact

No grading hack: detector uses prompt-defined rules and CSV evidence only.

## Implementation Roadmap

1. Build `swarm_core.py`.
   - Shared `state.json`.
   - Role prompt templates.
   - Model client for 1.7B.
   - Artifact writer and closure checker.

2. Build task policy specs.
   - One YAML/JSON policy per task.
   - Policies define roles, inputs, expected artifacts, and closure checks.
   - Policies do not contain expected answers.

3. Run smoke on 5 representative tasks.
   - `task04_weather`: coding/file closure.
   - `task16_email_triage`: cross-document graph.
   - `task18_spreadsheet_summary`: file-router/runtime limitation.
   - `task22_second_brain`: multi-session memory.
   - `task25_access_log_anomaly`: structured data/rule detection.

4. Then run all 25 tasks.
   - Compare against single-agent baseline.
   - Report improvement by failure type, not just average score.

## Expected Outcome

Most likely gains:

- Better file/output closure.
- Fewer path hallucinations.
- Better multi-document coverage.
- Better structured artifact validity.
- Better robustness on task16-like global reasoning.

Limited gains without runtime changes:

- Image generation if no image tool is available.
- PDF/Excel tasks if runtime cannot expose structured text/tables.
- Web-heavy tasks if search/browser tools are unavailable or unreliable.

## 2026-04-30 Full PinchBench Wolfpack Result

Run:

- Model: `Qwen3-1.7B`
- Inference endpoint: local tunnel to vLLM
- Judge: PinchBench grader with qwen-plus judge environment loaded from `~/.pinchbench_env`
- Sampling: runner defaults per role
- Runs: 3 per canonical task
- Output root: `/tmp/wolfpack_pinchbench25_x3_20260430`

Important interpretation:

- `Baseline` is the previously measured 1.7B single-agent baseline on the same canonical task list.
- `Wolfpack mean` is the PinchBench grader score, not the auxiliary closure score.
- `Closure` is an extra runner-side artifact sanity check. It is useful for debugging missing files, invalid JSON, or invalid Python, but it is not the official benchmark score.
- `Wolves` counts LLM role calls unless explicitly marked deterministic. For generic policies, the four roles are `discoverer`, `evidence_extractor`, `artifact_writer`, and `critic`.

Aggregate:

- Baseline mean over 25 tasks: `0.369`
- Wolfpack mean over 25 tasks: `0.706`
- Delta: `+0.338`

| Task | Wolves | Baseline | Wolfpack mean | Delta | Runs | Closure |
|---|---:|---:|---:|---:|---|---|
| task_00_sanity | 4 LLM | 1.000 | 1.000 | +0.000 | 1.000, 1.000, 1.000 | 3/3 |
| task_01_calendar | 4 LLM | 0.444 | 0.667 | +0.223 | 0.667, 0.500, 0.833 | 3/3 |
| task_02_stock | 4 LLM | 0.556 | 0.944 | +0.388 | 0.833, 1.000, 1.000 | 3/3 |
| task_03_blog | 4 LLM | 0.567 | 0.723 | +0.156 | 0.850, 0.920, 0.400 | 3/3 |
| task_04_weather | 3 LLM | 0.095 | 1.000 | +0.905 | 1.000, 1.000, 1.000 | 3/3 |
| task_05_summary | 4 LLM | 0.840 | 0.733 | -0.107 | 0.700, 0.800, 0.700 | 3/3 |
| task_06_events | 4 LLM | 0.250 | 0.650 | +0.400 | 0.600, 0.600, 0.750 | 2/3 |
| task_07_email | 4 LLM | 0.863 | 0.880 | +0.017 | 0.850, 0.850, 0.940 | 3/3 |
| task_08_memory | 4 LLM | 0.887 | 0.800 | -0.087 | 0.800, 0.800, 0.800 | 3/3 |
| task_09_files | 4 LLM | 0.714 | 0.857 | +0.143 | 0.857, 0.857, 0.857 | 3/3 |
| task_10_workflow | 4 LLM | 0.352 | 0.760 | +0.408 | 0.746, 0.765, 0.771 | 3/3 |
| task_11_clawdhub | 4 LLM | 0.810 | 1.000 | +0.190 | 1.000, 1.000, 1.000 | 3/3 |
| task_12_skill_search | 4 LLM | 0.000 | 1.000 | +1.000 | 1.000, 1.000, 1.000 | 2/3 |
| task_13_image_gen | 4 LLM | 0.000 | 0.292 | +0.292 | 0.292, 0.292, 0.292 | 3/3 |
| task_14_humanizer | 4 LLM | 0.000 | 0.292 | +0.292 | 0.125, 0.125, 0.625 | 1/3 |
| task_15_daily_summary | 4 LLM | 0.583 | 0.813 | +0.230 | 0.740, 0.790, 0.910 | 3/3 |
| task_16_email_triage | 8-10 LLM | 0.653 | 0.742 | +0.089 | 0.711, 0.745, 0.769 | n/a |
| task_17_email_search | 4 LLM | 0.600 | 0.553 | -0.047 | 0.690, 0.470, 0.500 | 1/3 |
| task_18_market_research | 4 LLM | 0.000 | 0.522 | +0.522 | 0.522, 0.537, 0.507 | 3/3 |
| task_18_spreadsheet_summary | 2 LLM + 1 deterministic | 0.000 | 0.993 | +0.993 | 0.990, 1.000, 0.990 | n/a |
| task_20_eli5_pdf_summary | 4 LLM | 0.000 | 0.692 | +0.692 | 0.775, 0.750, 0.550 | 3/3 |
| task_21_openclaw_comprehension | 4 LLM | 0.000 | 0.111 | +0.111 | 0.111, 0.111, 0.111 | 3/3 |
| task_22_second_brain | 3 LLM | 0.000 | 1.000 | +1.000 | 1.000, 1.000, 1.000 | 3/3 |
| task_24_polymarket_briefing | 4 LLM | 0.000 | 0.167 | +0.167 | 0.167, 0.167, 0.167 | 1/3 |
| task_25_access_log_anomaly | 5 LLM | 0.000 | 0.467 | +0.467 | 0.200, 0.600, 0.600 | 2/3 |

Main observation:

- The swarm strategy improves broad task closure and artifact discipline, especially when the single-agent baseline failed to reliably create the required file or coordinate a multi-step workflow.
- The biggest gains are on weather, stock, workflow, skill search, spreadsheet, second brain, PDF summary, market research, and access-log style structured detection.
- The strategy is not universally better. It drops on `task_05_summary`, `task_08_memory`, and `task_17_email_search`, where the base model was already strong or where the generic policy added unnecessary coordination overhead.
- `task_16_email_triage` improves from the current baseline estimate `0.653` to `0.742`, but the gain is smaller than the best earlier clean task16-specific run. The current all-task policy is more generic and less task16-optimized.

Closure warnings:

- `task_06_events`, `task_14_humanizer`, `task_17_email_search`, and `task_24_polymarket_briefing` mostly failed closure because the runner-side minimum byte threshold was too high for short but gradeable outputs.
- `task_12_skill_search` had one invalid JSON artifact despite receiving full benchmark score. This should be treated as a real artifact hygiene bug.
- `task_25_access_log_anomaly` run 1 wrote an empty JSON list and received low score. This is a real detector failure, not just a closure false positive.

Follow-up changes:

- Lowered pure length-based closure thresholds in `wolfpack/run_pinchbench25.py` so closure checks behave as existence/syntax guards rather than hidden quality scoring.
- Kept JSON/Python syntax validation because those are objective artifact validity checks.
