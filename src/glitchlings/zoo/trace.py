"""Execution tracing for observable per-stage glitchling orchestration.

Trace mode records one entry per glitchling execution stage in real execution
order. The recorder is opt-in and stateful, so this module lives on the
impure side of the functional purity architecture; all validation it relies
on is delegated to the pure :mod:`glitchlings.zoo.validation` boundary.

The recorder never participates in execution decisions: traced runs use the
same plans, RNGs, Rust pipelines, and fallback calls as ordinary runs, which
keeps the final corrupted output identical to a non-traced run.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import yaml

from ..constants import (
    DEFAULT_TRACE_PREVIEW_LIMIT,
    TRACE_FORMAT_JSON,
    TRACE_FORMAT_YAML,
)
from .validation import normalize_trace_format, validate_trace_preview_limit

# Execution path labels recorded for every stage.
TRACE_PATH_PIPELINE = "rust_pipeline"
TRACE_PATH_FALLBACK = "python_fallback"

# Stage and overall trace status labels.
STATUS_OK = "ok"
STATUS_FAILED = "failed"

HASH_ALGORITHM = "sha256"


# ---------------------------------------------------------------------------
# Pure observation helpers
# ---------------------------------------------------------------------------


def text_sha256(text: str) -> str:
    """Return the SHA-256 hex digest of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_changed_chars(before: str, after: str) -> int:
    """Count characters changed between two strings.

    The count is derived from :class:`difflib.SequenceMatcher` opcodes:
    deletions and insertions contribute their span length, while
    replacements contribute the longer of the removed/inserted spans. Equal
    spans contribute nothing. This is a deterministic, dependency-free
    character-level edit estimate rather than a full Levenshtein distance.

    Args:
        before: Text entering the stage.
        after: Text leaving the stage.

    Returns:
        The number of characters accounted for by non-equal edit spans.
    """
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "delete":
            changed += i2 - i1
        elif tag == "insert":
            changed += j2 - j1
        else:  # "replace"
            changed += max(i2 - i1, j2 - j1)
    return changed


def build_preview(text: str, limit: int) -> tuple[str, bool]:
    """Return a capped preview and whether truncation occurred.

    Args:
        text: Source text.
        limit: Maximum number of characters to keep.

    Returns:
        A ``(preview, truncated)`` tuple.
    """
    truncated = len(text) > limit
    return text[:limit], truncated


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Validated configuration for a trace recording.

    Attributes:
        preview_limit: Maximum number of characters kept in each text
            preview. Validated against the server-side hard cap at
            construction time so invalid values fail before execution starts.
    """

    preview_limit: int = DEFAULT_TRACE_PREVIEW_LIMIT

    def __post_init__(self) -> None:
        validated = validate_trace_preview_limit(self.preview_limit)
        object.__setattr__(self, "preview_limit", validated)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class TraceRecorder:
    """Accumulates ordered stage records for a single traced corruption run."""

    def __init__(
        self,
        config: TraceConfig | None = None,
        *,
        master_seed: int | None = None,
    ) -> None:
        """Initialize the recorder.

        Args:
            config: Optional trace configuration. A default config is built
                when omitted.
            master_seed: Master seed of the Gaggle being observed.
        """
        self.config: TraceConfig = config if config is not None else TraceConfig()
        self.master_seed: int | None = master_seed
        self.stages: list[dict[str, Any]] = []
        self.error: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        """Return the overall trace status."""
        return STATUS_FAILED if self.error is not None else STATUS_OK

    @property
    def completed_stages(self) -> int:
        """Return the number of stages that finished successfully."""
        return sum(1 for stage in self.stages if stage["status"] == STATUS_OK)

    def _preview_fields(self, prefix: str, text: str) -> dict[str, Any]:
        preview, truncated = build_preview(text, self.config.preview_limit)
        return {
            f"{prefix}_preview": preview,
            f"{prefix}_preview_truncated": truncated,
        }

    def record_stage(
        self,
        *,
        name: str,
        seed: int,
        scope: str,
        order: str,
        execution_path: str,
        input_text: str,
        output_text: str,
    ) -> dict[str, Any]:
        """Record a successfully completed stage.

        Returns:
            The stored stage record.
        """
        record: dict[str, Any] = {
            "index": len(self.stages),
            "name": name,
            "seed": int(seed),
            "scope": scope,
            "order": order,
            "execution_path": execution_path,
            "status": STATUS_OK,
            "input_length": len(input_text),
            "output_length": len(output_text),
            "input_hash": text_sha256(input_text),
            "output_hash": text_sha256(output_text),
            "changed_chars": count_changed_chars(input_text, output_text),
        }
        record.update(self._preview_fields("input", input_text))
        record.update(self._preview_fields("output", output_text))
        self.stages.append(record)
        return record

    def record_failure(
        self,
        *,
        name: str,
        seed: int | None,
        scope: str,
        order: str,
        execution_path: str,
        input_text: str,
        error: BaseException,
    ) -> dict[str, Any]:
        """Record a stage that raised, preserving the prefix results.

        The exception is only observed here; callers remain responsible for
        re-raising it so failures are never swallowed.

        Returns:
            The stored failure record.
        """
        stage_index = len(self.stages)
        error_payload = {
            "type": type(error).__name__,
            "message": str(error),
        }
        record: dict[str, Any] = {
            "index": stage_index,
            "name": name,
            "seed": None if seed is None else int(seed),
            "scope": scope,
            "order": order,
            "execution_path": execution_path,
            "status": STATUS_FAILED,
            "input_length": len(input_text),
            "output_length": None,
            "input_hash": text_sha256(input_text),
            "output_hash": None,
            "changed_chars": None,
            "output_preview": None,
            "output_preview_truncated": None,
            "error": error_payload,
        }
        record.update(self._preview_fields("input", input_text))
        self.stages.append(record)
        self.error = {
            "stage_index": stage_index,
            "stage_name": name,
            "execution_path": execution_path,
            "completed_stages": stage_index,
            **error_payload,
        }
        return record

    def to_payload(self) -> dict[str, Any]:
        """Return the trace as a JSON/YAML-serializable payload."""
        return {
            "status": self.status,
            "hash_algorithm": HASH_ALGORITHM,
            "master_seed": self.master_seed,
            "preview_limit": self.config.preview_limit,
            "stage_count": len(self.stages),
            "stages": list(self.stages),
            "error": self.error,
        }

    def render(self, output_format: str = TRACE_FORMAT_JSON) -> str:
        """Serialize the trace payload in the requested format.

        Args:
            output_format: "json", "yaml", or the "yml" alias.

        Returns:
            The serialized trace document.
        """
        normalized = normalize_trace_format(output_format)
        payload = self.to_payload()
        if normalized == TRACE_FORMAT_YAML:
            return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        return json.dumps(payload, indent=2, ensure_ascii=False)


__all__ = [
    "TRACE_PATH_PIPELINE",
    "TRACE_PATH_FALLBACK",
    "STATUS_OK",
    "STATUS_FAILED",
    "HASH_ALGORITHM",
    "TraceConfig",
    "TraceRecorder",
    "text_sha256",
    "count_changed_chars",
    "build_preview",
]
