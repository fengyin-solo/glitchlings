"""Impure execution dispatch for Gaggle orchestration.

This module handles the actual execution of glitchling plans, including
Rust FFI calls and Python fallback execution. It is the impure counterpart
to core_planning.py.

**Design Philosophy:**

This module is explicitly *impure* - it invokes compiled Rust functions
and executes Python corruption functions with side effects. All Rust
dispatch for Gaggle orchestration flows through this module.

The separation allows:
- Pure planning logic to be tested without Rust
- Clear boundaries between plan construction and execution
- Mocking execution for integration tests

See AGENTS.md "Functional Purity Architecture" for full details.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from glitchlings.internal.rust_ffi import (
    compose_operations_rust,
    compose_operations_staged_rust,
)
from glitchlings.zoo.trace import (
    TRACE_PATH_FALLBACK,
    TRACE_PATH_PIPELINE,
    TraceRecorder,
)

if TYPE_CHECKING:
    from .core_planning import ExecutionPlan, GlitchlingProtocol, PipelineDescriptor


# ---------------------------------------------------------------------------
# Plan Execution
# ---------------------------------------------------------------------------


def execute_plan(
    text: str,
    plan: ExecutionPlan,
    master_seed: int,
    *,
    include_only_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    pipeline: Any | None = None,
) -> str:
    """Execute an orchestration plan against input text.

    This function dispatches plan steps to either the Rust pipeline or
    Python fallback execution as appropriate.

    Args:
        text: Input text to transform.
        plan: Execution plan from build_execution_plan().
        master_seed: Master seed for Rust pipeline.

    Returns:
        Transformed text after all plan steps complete.
    """
    # Fast path: all glitchlings support pipeline
    if plan.all_pipeline and plan.step_count == 1:
        if pipeline is not None:
            return compose_with_pipeline(
                pipeline,
                text,
                include_only_patterns=include_only_patterns,
                exclude_patterns=exclude_patterns,
            )
        descriptors = list(plan.steps[0].descriptors)
        return compose_operations_rust(
            text,
            descriptors,
            master_seed,
            include_only_patterns=include_only_patterns,
            exclude_patterns=exclude_patterns,
        )

    # Hybrid path: mix of pipeline batches and individual fallbacks
    result = text

    for step in plan.steps:
        if step.is_pipeline_step:
            # Execute the batch through the Rust pipeline
            descriptors = list(step.descriptors)
            result = compose_operations_rust(
                result,
                descriptors,
                master_seed,
                include_only_patterns=include_only_patterns,
                exclude_patterns=exclude_patterns,
            )
        elif step.fallback_glitchling is not None:
            # Execute single glitchling via Python fallback
            result = cast(str, step.fallback_glitchling.corrupt(result))

    return result


def execute_descriptors(
    text: str,
    descriptors: list[PipelineDescriptor],
    master_seed: int,
    *,
    include_only_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    pipeline: Any | None = None,
) -> str:
    """Execute a list of pipeline descriptors through Rust.

    This is a thin wrapper over compose_operations_rust for cases
    where the caller has already constructed descriptors directly.

    Args:
        text: Input text to transform.
        descriptors: Pipeline descriptors for each glitchling.
        master_seed: Master seed for Rust pipeline.

    Returns:
        Transformed text.
    """
    if pipeline is not None:
        return compose_with_pipeline(
            pipeline,
            text,
            include_only_patterns=include_only_patterns,
            exclude_patterns=exclude_patterns,
        )

    return compose_operations_rust(
        text,
        descriptors,
        master_seed,
        include_only_patterns=include_only_patterns,
        exclude_patterns=exclude_patterns,
    )


def compose_with_pipeline(
    pipeline: Any,
    text: str,
    *,
    include_only_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> str:
    """Run a pre-built Rust pipeline while preserving masking kwargs."""
    if include_only_patterns or exclude_patterns:
        raise RuntimeError(
            "Cached pipelines already embed masking patterns; overrides are not supported."
        )
    return cast(str, pipeline.run(text))


# ---------------------------------------------------------------------------
# Traced plan execution
# ---------------------------------------------------------------------------


def _stage_scope_order(glitchling: GlitchlingProtocol) -> tuple[str, str]:
    """Return the lowercase enum names used in trace records."""
    return glitchling.level.name.lower(), glitchling.order.name.lower()


def _raise_staged_failure(kind: str, message: str) -> None:
    """Raise the native exception type matching the Rust failure kind."""
    if kind == "RuntimeError":
        raise RuntimeError(message)
    raise ValueError(message)


def execute_plan_traced(
    text: str,
    plan: ExecutionPlan,
    members: Sequence[GlitchlingProtocol],
    master_seed: int,
    recorder: TraceRecorder,
    *,
    include_only_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    pipeline: Any | None = None,
) -> str:
    """Execute a plan while recording one observable stage per glitchling.

    Pipeline batches are split into per-operation snapshots by the Rust
    ``run_staged`` entry point, which keeps the same buffer, RNGs, and order
    as a normal batch run. Fallback steps are executed individually through
    Python. In both cases stages are recorded in true execution order and
    failures are marked on the recorder before being re-raised.

    Args:
        text: Input text to transform.
        plan: Execution plan from build_execution_plan().
        members: Glitchlings aligned with the plan, in execution order.
        master_seed: Master seed for Rust pipeline construction.
        recorder: Recorder collecting per-stage observations.
        include_only_patterns: Include masks applied to pipeline steps.
        exclude_patterns: Exclude masks applied to pipeline steps.
        pipeline: Optional pre-built Rust pipeline for the all-pipeline case.

    Returns:
        Transformed text after all plan steps complete.
    """
    result = text
    cursor = 0

    for step in plan.steps:
        if step.is_pipeline_step:
            descriptors = list(step.descriptors)
            batch_members = list(members[cursor : cursor + len(descriptors)])
            if pipeline is not None:
                snapshots, failure = pipeline.run_staged(result)
            else:
                snapshots, failure = compose_operations_staged_rust(
                    result,
                    descriptors,
                    master_seed,
                    include_only_patterns=include_only_patterns,
                    exclude_patterns=exclude_patterns,
                )

            for offset, snapshot in enumerate(snapshots):
                member = batch_members[offset]
                scope, order = _stage_scope_order(member)
                stage_input = result if offset == 0 else snapshots[offset - 1]
                recorder.record_stage(
                    name=member.name,
                    seed=int(descriptors[offset]["seed"]),
                    scope=scope,
                    order=order,
                    execution_path=TRACE_PATH_PIPELINE,
                    input_text=stage_input,
                    output_text=snapshot,
                )

            if failure is not None:
                failed_name, kind, message = failure
                failed_offset = len(snapshots)
                member = batch_members[failed_offset]
                scope, order = _stage_scope_order(member)
                stage_input = result if failed_offset == 0 else snapshots[-1]
                error = RuntimeError(message) if kind == "RuntimeError" else ValueError(message)
                recorder.record_failure(
                    name=failed_name,
                    seed=int(descriptors[failed_offset]["seed"]),
                    scope=scope,
                    order=order,
                    execution_path=TRACE_PATH_PIPELINE,
                    input_text=stage_input,
                    error=error,
                )
                _raise_staged_failure(kind, message)

            result = snapshots[-1]
            cursor += len(descriptors)
            continue

        fallback_member = step.fallback_glitchling
        if fallback_member is None:  # pragma: no cover - planner guarantees one branch
            message = "Execution step has neither pipeline descriptors nor a fallback glitchling"
            raise RuntimeError(message)
        fallback_seed = fallback_member.seed
        if fallback_seed is None:  # pragma: no cover - Gaggle always derives seeds
            message = f"Fallback glitchling {fallback_member.name} executed without a derived seed"
            raise RuntimeError(message)
        scope, order = _stage_scope_order(fallback_member)
        try:
            output = cast(str, fallback_member.corrupt(result))
        except Exception as exc:
            recorder.record_failure(
                name=fallback_member.name,
                seed=fallback_seed,
                scope=scope,
                order=order,
                execution_path=TRACE_PATH_FALLBACK,
                input_text=result,
                error=exc,
            )
            raise
        recorder.record_stage(
            name=fallback_member.name,
            seed=int(fallback_seed),
            scope=scope,
            order=order,
            execution_path=TRACE_PATH_FALLBACK,
            input_text=result,
            output_text=output,
        )
        result = output
        cursor += 1

    return result


__all__ = [
    "execute_plan",
    "execute_descriptors",
    "execute_plan_traced",
]
