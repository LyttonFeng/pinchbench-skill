# task16 Reward Trim Plan

Date: 2026-04-23  
Task: `task_16_email_triage`

## Goal

Keep reward logic:

1. stable
2. verifiable
3. hard to hack

Do **not** keep adding fine-grained semantic reward terms just to chase benchmark misses.  
Those misses should mainly be addressed with targeted data.

## Current diagnosis

The current `task16 v2` reward already helped with the lower-level behavior:

1. reduce repeated inbox rereads
2. improve report completion
3. improve structured output

But the remaining benchmark errors are now more semantic:

1. `email_13` correlated alert is omitted or downgraded
2. `email_01` and `email_13` are not reliably linked
3. `email_05` BigClient is sometimes missed or under-prioritized
4. `email_08` security/compliance is not reliably prioritized high enough

These are exactly the kind of errors that are risky to encode directly into reward.

## Keep

These reward terms should remain because they are process-level and easy to verify.

### 1. `bulk_reread_loop penalty`

Reason:

1. directly targets a real observed failure mode
2. easy to detect from tool calls
3. difficult to game without actually changing behavior

### 2. `late_no_report penalty`

Reason:

1. directly targets "read forever, never finish"
2. aligned with task completion
3. stable across prompt variants

### 3. `report_created reward`

Reason:

1. weak but useful completion signal
2. file creation is easy to verify

### 4. `structured_complete_report reward`

Reason:

1. pushes the model toward a usable artifact
2. still mostly structural, not semantic guessing
3. more robust than rewarding keyword mentions

### 5. `context_overflow penalty`

Reason:

1. directly punishes pathological long-loop behavior
2. highly verifiable

### 6. terminal reward

Reason:

1. remains the cleanest end objective
2. keeps optimization anchored on final task outcome

## Trim

These terms should be weakened or removed because they are too easy to satisfy by mention-level behavior.

### 1. `incident_linkage`

Problem:

1. current implementation can reward loose textual mention
2. does not verify that the linkage is represented correctly in the final report

Decision:

- lower weight sharply or remove

### 2. `bigclient_high_priority`

Problem:

1. mentioning `BigClient` is not the same as ranking it correctly
2. easy to hack with superficial wording

Decision:

- lower weight sharply or remove

### 3. `security_high_priority`

Problem:

1. mentioning `security` or `compliance` is not enough
2. does not ensure correct priority or action

Decision:

- lower weight sharply or remove

## Principle going forward

Use reward for:

1. process control
2. artifact completion
3. obvious bad-loop suppression

Use data for:

1. which emails matter most
2. which emails must be linked
3. what correct priority looks like
4. what correct action looks like

## Recommended next action

Do **not** build `task16 reward v3` around fine-grained email-level semantics.

Instead:

1. trim the semantic reward terms
2. keep the stable process-control core
3. move the next wave of improvement into targeted data
