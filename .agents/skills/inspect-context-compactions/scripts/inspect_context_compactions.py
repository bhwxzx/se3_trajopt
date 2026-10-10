#!/usr/bin/env python3
"""Inspect one Codex rollout and report a verified compaction count."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLD = 5
DEFAULT_STABILITY_RETRIES = 2
DEFAULT_RETRY_DELAY_SECONDS = 0.05
THREAD_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def _default_sessions_root() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "sessions"
    return Path.home() / ".codex" / "sessions"


def _base_result(thread_id: str | None, threshold: int) -> dict[str, Any]:
    return {
        "version": 2,
        "provider": "codex",
        "status": "unavailable",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "rollout_path": None,
        "compaction_count": None,
        "window_numbers": [],
        "compaction_timestamps": [],
        "threshold": threshold,
        "threshold_reached": None,
        "event_cross_check": {
            "context_compacted_count": None,
            "completion_event_count": None,
            "unique_completion_count": None,
            "duplicate_completion_count": None,
            "matched_window_count": None,
            "matches": None,
        },
        "errors": [],
    }


def _resolve_rollout(
    sessions_root: Path,
    thread_id: str | None,
    rollout_path: Path | None,
) -> tuple[Path | None, list[str]]:
    if rollout_path is not None:
        resolved = rollout_path.expanduser().resolve()
        if not resolved.is_file():
            return None, ["rollout_not_found"]
        return resolved, []
    if not thread_id:
        return None, ["codex_thread_id_unavailable"]
    if THREAD_ID_PATTERN.fullmatch(thread_id) is None:
        return None, ["invalid_thread_id"]
    root = sessions_root.expanduser().resolve()
    if not root.is_dir():
        return None, ["sessions_root_not_found"]
    matches = sorted(root.rglob(f"rollout-*-{thread_id}.jsonl"))
    if not matches:
        return None, ["rollout_not_found"]
    if len(matches) != 1:
        return None, ["multiple_rollouts_match_thread_id"]
    return matches[0].resolve(), []


def _rollout_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_size, stat.st_mtime_ns


def _has_retryable_tail_error(report: dict[str, Any]) -> bool:
    errors = report.get("errors")
    if not isinstance(errors, list):
        return False
    return any(
        error == "context_compacted_event_count_mismatch"
        or (isinstance(error, str) and error.startswith("invalid_json_line:"))
        for error in errors
    )


def _inspect_rollout_once(
    path: Path,
    thread_id: str | None,
    threshold: int,
) -> dict[str, Any]:
    result = _base_result(thread_id, threshold)
    result["rollout_path"] = str(path)
    errors: list[str] = []
    session_ids: set[str] = set()
    window_numbers: list[int] = []
    compaction_timestamps: list[str | None] = []
    context_compacted_count = 0
    completion_event_count = 0
    duplicate_completion_count = 0
    # Notifications are evidence for the preceding compacted record, never
    # additional compactions. Keep each format separate within that window.
    legacy_windows: set[int] = set()
    completion_windows: set[int] = set()
    completion_ids: dict[str, str] = {}
    event_thread_ids: set[str] = set()

    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    errors.append(f"invalid_json_line:{line_number}")
                    continue
                if not isinstance(record, dict):
                    errors.append(f"non_object_record:{line_number}")
                    continue
                record_type = record.get("type")
                payload = record.get("payload")
                if record_type == "session_meta" and isinstance(payload, dict):
                    session_id = payload.get("id") or payload.get("session_id")
                    if isinstance(session_id, str) and session_id:
                        session_ids.add(session_id)
                elif record_type == "compacted":
                    if not isinstance(payload, dict):
                        errors.append(f"invalid_compacted_payload:{line_number}")
                        continue
                    window_number = payload.get("window_number")
                    if (
                        isinstance(window_number, bool)
                        or not isinstance(window_number, int)
                        or window_number < 1
                    ):
                        errors.append(f"invalid_window_number:{line_number}")
                        continue
                    window_numbers.append(window_number)
                    timestamp = record.get("timestamp")
                    compaction_timestamps.append(
                        timestamp if isinstance(timestamp, str) else None
                    )
                elif (
                    record_type == "event_msg"
                    and isinstance(payload, dict)
                ):
                    event_type = payload.get("type")
                    item = payload.get("item")
                    is_completion = (
                        event_type == "item_completed"
                        and isinstance(item, dict)
                        and item.get("type") in (
                            "ContextCompaction", "contextCompaction"
                        )
                    )
                    if event_type != "context_compacted" and not is_completion:
                        continue
                    event_thread = payload.get("thread_id")
                    if "thread_id" in payload:
                        if not isinstance(event_thread, str) or not event_thread:
                            errors.append(f"invalid_compaction_event_thread:{line_number}")
                        else:
                            event_thread_ids.add(event_thread)
                    window = len(window_numbers)
                    if is_completion:
                        completion_event_count += 1
                        item_id = item.get("id")
                        if not isinstance(item_id, str) or not item_id.strip():
                            errors.append(f"invalid_compaction_item_id:{line_number}")
                            continue
                        # Outer log timestamps may differ on replay. Compare the
                        # event payload, normalizing the two item-type spellings.
                        normalized = dict(payload)
                        normalized["item"] = dict(item, type="ContextCompaction")
                        fingerprint = json.dumps(normalized, sort_keys=True)
                        if item_id in completion_ids:
                            if completion_ids[item_id] != fingerprint:
                                errors.append(f"conflicting_compaction_item:{line_number}")
                            else:
                                duplicate_completion_count += 1
                            continue
                        completion_ids[item_id] = fingerprint
                        if window in completion_windows:
                            errors.append(f"multiple_compaction_items_for_window:{line_number}")
                        completion_windows.add(window)
                    else:
                        context_compacted_count += 1
                        if window in legacy_windows:
                            # Legacy events have no reliable item identity;
                            # identical text/timestamps do not prove a replay.
                            errors.append(f"ambiguous_legacy_compaction_events:{line_number}")
                        legacy_windows.add(window)
                    if window == 0:
                        errors.append(f"compaction_event_without_window:{line_number}")
    except OSError:
        result["errors"] = ["rollout_read_failed"]
        return result

    if not session_ids:
        errors.append("session_meta_missing")
    elif len(session_ids) != 1:
        errors.append("multiple_session_ids_in_rollout")
    else:
        session_id = next(iter(session_ids))
        if thread_id and thread_id != session_id:
            errors.append("rollout_thread_id_mismatch")
        elif not thread_id:
            result["thread_id"] = session_id
        if event_thread_ids - {session_id}:
            errors.append("compaction_event_thread_id_mismatch")

    expected_windows = list(range(1, len(window_numbers) + 1))
    if window_numbers != expected_windows:
        errors.append("compaction_window_sequence_invalid")
    matched_windows = legacy_windows | completion_windows
    event_matches = matched_windows == set(expected_windows)
    if not event_matches:
        errors.append("context_compacted_event_count_mismatch")

    result["window_numbers"] = window_numbers
    result["compaction_timestamps"] = compaction_timestamps
    result["event_cross_check"] = {
        "context_compacted_count": context_compacted_count,
        "completion_event_count": completion_event_count,
        "unique_completion_count": len(completion_ids),
        "duplicate_completion_count": duplicate_completion_count,
        "matched_window_count": len(matched_windows - {0}),
        "matches": event_matches,
    }
    result["errors"] = errors
    if errors:
        result["status"] = "inconsistent"
        return result

    count = len(window_numbers)
    result["status"] = "available"
    result["compaction_count"] = count
    result["threshold_reached"] = count >= threshold
    return result


def _inspect_rollout(
    path: Path,
    thread_id: str | None,
    threshold: int,
    *,
    stability_retries: int,
    retry_delay_seconds: float,
) -> dict[str, Any]:
    last_result: dict[str, Any] | None = None
    changed_during_last_read = False
    for attempt in range(stability_retries + 1):
        signature_before = _rollout_signature(path)
        last_result = _inspect_rollout_once(path, thread_id, threshold)
        signature_after = _rollout_signature(path)
        changed_during_last_read = signature_before != signature_after
        should_retry = changed_during_last_read or _has_retryable_tail_error(
            last_result
        )
        if not should_retry:
            return last_result
        if attempt < stability_retries and retry_delay_seconds:
            time.sleep(retry_delay_seconds)

    assert last_result is not None
    if changed_during_last_read:
        result = _base_result(thread_id, threshold)
        result["rollout_path"] = str(path)
        result["errors"] = ["rollout_changed_during_read"]
        return result
    return last_result


def inspect_context_compactions(
    *,
    sessions_root: Path,
    threshold: int = DEFAULT_THRESHOLD,
    thread_id: str | None = None,
    rollout_path: Path | None = None,
    stability_retries: int = DEFAULT_STABILITY_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> dict[str, Any]:
    """Return a content-free compaction count for one exact Codex thread."""
    if threshold < 1:
        raise ValueError("threshold must be a positive integer")
    if stability_retries < 0:
        raise ValueError("stability_retries must be non-negative")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    resolved, errors = _resolve_rollout(sessions_root, thread_id, rollout_path)
    if resolved is None:
        result = _base_result(thread_id, threshold)
        result["errors"] = errors
        return result
    return _inspect_rollout(
        resolved,
        thread_id,
        threshold,
        stability_retries=stability_retries,
        retry_delay_seconds=retry_delay_seconds,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id")
    parser.add_argument("--sessions-root", default=str(_default_sessions_root()))
    parser.add_argument("--rollout")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--stability-retries",
        type=int,
        default=DEFAULT_STABILITY_RETRIES,
    )
    parser.add_argument(
        "--retry-delay-seconds",
        type=float,
        default=DEFAULT_RETRY_DELAY_SECONDS,
    )
    args = parser.parse_args()
    if args.threshold < 1:
        parser.error("--threshold must be a positive integer")
    if args.stability_retries < 0:
        parser.error("--stability-retries must be non-negative")
    if args.retry_delay_seconds < 0:
        parser.error("--retry-delay-seconds must be non-negative")

    thread_id = args.thread_id
    if thread_id is None and args.rollout is None:
        thread_id = os.environ.get("CODEX_THREAD_ID")
    report = inspect_context_compactions(
        sessions_root=Path(args.sessions_root),
        threshold=args.threshold,
        thread_id=thread_id,
        rollout_path=Path(args.rollout) if args.rollout else None,
        stability_retries=args.stability_retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
