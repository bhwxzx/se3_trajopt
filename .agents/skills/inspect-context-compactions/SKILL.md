---
name: inspect-context-compactions
description: Reliably inspect Codex rollout JSONL to determine an exact, content-free context-compaction count for the current or explicitly selected session. Use when Codex must check whether a session has been compacted repeatedly, apply a compaction threshold, prepare a trustworthy session handoff, or audit compaction metadata without exposing conversation or replacement-history content.
---

# Inspect Context Compactions

Use the bundled read-only inspector instead of estimating compactions from
summaries, token usage, event prose, or conversation length.

## Run the inspector

From the repository root, run:

```bash
python3 \
  .agents/skills/inspect-context-compactions/scripts/inspect_context_compactions.py
```

The default invocation reads `CODEX_THREAD_ID`, searches the active Codex
sessions directory, and uses a threshold of five. Override inputs only when the
current runtime does not expose them:

```bash
python3 .agents/skills/inspect-context-compactions/scripts/inspect_context_compactions.py \
  --thread-id <thread-id> \
  --sessions-root <sessions-directory> \
  --threshold <positive-integer>

python3 .agents/skills/inspect-context-compactions/scripts/inspect_context_compactions.py \
  --rollout <exact-rollout.jsonl> \
  --threshold <positive-integer>
```

Prefer `--rollout` only when the exact file is already known. The default
sessions directory is `$CODEX_HOME/sessions` when `CODEX_HOME` is set and
`~/.codex/sessions` otherwise. When `--rollout` is provided without an explicit
`--thread-id`, ignore `CODEX_THREAD_ID` and recover the identity from that
rollout. Add `--thread-id` only when the caller wants an identity cross-check.

The inspector retries a changing file or a possibly partial tail twice with a
short delay. Override this bounded behavior only for diagnostics with
`--stability-retries` and `--retry-delay-seconds`.

## Interpret the report

Use the JSON `status` before reading the count:

- `available`: Trust `compaction_count` and `threshold_reached`.
- `unavailable`: Report the exact count as unavailable; do not infer one.
- `inconsistent`: Report that the rollout metadata failed validation; do not
  use the partial record count as an exact count.

Treat `errors` as stable machine-readable reason codes. Use `window_numbers`,
timestamps, and the mirrored-event cross-check only as metadata evidence.

## Preserve counting and privacy guarantees

- Count only top-level `compacted` records.
- Require consecutive `window_number` values starting at one.
- Cross-check top-level `event_msg` records with payload type
  `context_compacted`, or `item_completed` whose `item.type` is
  `ContextCompaction` / `contextCompaction`. Ignore start events and nested
  conversation content; never add notifications to the compaction count.
- Associate notifications with the preceding top-level `compacted` record.
  Each window needs at least one supported notification. Old and new formats
  may coexist in a window or vary between windows; do not sum their counts.
- Deduplicate completed items by nonempty `item.id` within the rollout, even
  when an identical event is replayed after a later window. Compare payloads
  with item-type spelling normalized; ignore the outer log timestamp.
  Conflicting payloads for one ID, distinct completion IDs in one window,
  missing IDs, notifications before any window, and mismatched event thread
  identities fail validation. Repeated legacy notifications in one window
  are ambiguous without a reliable ID and also fail validation.
- Report version 2 keeps `context_compacted_count` as the raw legacy count;
  `completion_event_count`, `unique_completion_count`,
  `duplicate_completion_count`, and `matched_window_count` expose the modern
  cross-check. Always use the overall `status` to decide whether to trust it.
- Require one rollout session identity and match it to the requested thread.
- Resolve exactly one rollout for automatic thread lookup.
- Return `rollout_changed_during_read` instead of claiming an exact count when
  the file keeps changing across all bounded attempts.
- Never print messages, tool output, `replacement_history`, or other rollout
  content.
- Keep the operation read-only. Do not rewrite, truncate, move, or delete
  rollout files.

If a workflow has its own handoff threshold or fallback policy, apply that
policy only after reading this inspector's verified status.
