"""Tests for core_planning.py pure functions.

These tests verify the pure planning functions work correctly without
Rust dependencies. The planning logic should be fully testable with
mock glitchlings that implement the protocol.
"""

from __future__ import annotations

from typing import Any

import pytest

from glitchlings.zoo.core_planning import (
    ExecutionPlan,
    ExecutionStep,
    NormalizedPlanSpec,
    PipelineDescriptor,
    PipelineDescriptorModel,
    PipelineOperationPayload,
    build_execution_plan,
    build_pipeline_descriptor,
    build_plan_preview,
    extract_plan_ordering,
    extract_plan_seeds,
    normalize_plan_entries,
    normalize_plan_specs,
    validate_plan_coverage,
)

# ---------------------------------------------------------------------------
# Mock Glitchling for Testing
# ---------------------------------------------------------------------------


class MockGlitchling:
    """A lightweight mock glitchling for testing planning functions.

    This mock implements the GlitchlingProtocol without any Rust dependencies,
    allowing pure planning tests.
    """

    def __init__(
        self,
        name: str,
        level: int,
        order: int,
        *,
        seed: int | None = None,
        has_pipeline: bool = True,
        gaggle_index: int | None = None,
    ) -> None:
        self.name = name
        self.level = level
        self.order = order
        self.seed = seed
        self._has_pipeline = has_pipeline
        if gaggle_index is not None:
            self._gaggle_index = gaggle_index

    def pipeline_operation(self) -> PipelineOperationPayload | None:
        if not self._has_pipeline:
            return None
        return {"type": f"mock_{self.name.lower()}"}

    def corrupt(self, text: str) -> str:
        return f"[{self.name}]{text}"


def mock_derive_seed(master_seed: int, name: str, index: int) -> int:
    """Simple deterministic seed derivation for testing."""
    return (master_seed + hash(name) + index) % (2**32)


# ---------------------------------------------------------------------------
# NormalizedPlanSpec Tests
# ---------------------------------------------------------------------------


class TestNormalizedPlanSpec:
    """Tests for NormalizedPlanSpec dataclass."""

    def test_from_glitchling(self) -> None:
        glitchling = MockGlitchling("TestGlitch", level=3, order=2)
        spec = NormalizedPlanSpec.from_glitchling(glitchling)

        assert spec.name == "TestGlitch"
        assert spec.scope == 3
        assert spec.order == 2

    def test_from_mapping(self) -> None:
        mapping = {"name": "MapGlitch", "scope": 4, "order": 1}
        spec = NormalizedPlanSpec.from_mapping(mapping)

        assert spec.name == "MapGlitch"
        assert spec.scope == 4
        assert spec.order == 1

    def test_from_mapping_coerces_types(self) -> None:
        # Strings that look like ints should be coerced
        mapping: dict[str, Any] = {"name": 123, "scope": "4", "order": "1"}
        spec = NormalizedPlanSpec.from_mapping(mapping)

        assert spec.name == "123"
        assert spec.scope == 4
        assert spec.order == 1

    def test_from_mapping_missing_field_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required field"):
            NormalizedPlanSpec.from_mapping({"name": "Test", "scope": 1})

    def test_from_entry_with_glitchling(self) -> None:
        glitchling = MockGlitchling("EntryGlitch", level=2, order=3)
        spec = NormalizedPlanSpec.from_entry(glitchling)

        assert spec.name == "EntryGlitch"

    def test_from_entry_with_mapping(self) -> None:
        mapping = {"name": "EntryMap", "scope": 1, "order": 5}
        spec = NormalizedPlanSpec.from_entry(mapping)

        assert spec.name == "EntryMap"

    def test_from_entry_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected Glitchling instances"):
            NormalizedPlanSpec.from_entry("not_a_glitchling")  # type: ignore[arg-type]

    def test_as_mapping(self) -> None:
        spec = NormalizedPlanSpec("AsMapTest", 3, 4)
        mapping = spec.as_mapping()

        assert mapping == {"name": "AsMapTest", "scope": 3, "order": 4}


class TestNormalizeFunctions:
    """Tests for normalize_plan_entries and normalize_plan_specs."""

    def test_normalize_plan_entries(self) -> None:
        glitchlings = [
            MockGlitchling("A", level=1, order=1),
            MockGlitchling("B", level=2, order=2),
        ]
        specs = normalize_plan_entries(glitchlings)

        assert len(specs) == 2
        assert specs[0].name == "A"
        assert specs[1].name == "B"

    def test_normalize_plan_specs(self) -> None:
        raw_specs = [
            {"name": "X", "scope": 1, "order": 1},
            {"name": "Y", "scope": 2, "order": 2},
        ]
        specs = normalize_plan_specs(raw_specs)

        assert len(specs) == 2
        assert specs[0].name == "X"
        assert specs[1].name == "Y"


# ---------------------------------------------------------------------------
# PipelineDescriptorModel Tests
# ---------------------------------------------------------------------------


class TestPipelineDescriptorModel:
    """Tests for PipelineDescriptorModel dataclass."""

    def test_as_mapping(self) -> None:
        model = PipelineDescriptorModel(
            name="TestPipeline",
            seed=12345,
            operation={"type": "test_op"},
        )
        mapping = model.as_mapping()

        assert mapping["name"] == "TestPipeline"
        assert mapping["seed"] == 12345
        assert mapping["operation"].get("type") == "test_op"


# ---------------------------------------------------------------------------
# build_pipeline_descriptor Tests
# ---------------------------------------------------------------------------


class TestBuildPipelineDescriptor:
    """Tests for build_pipeline_descriptor function."""

    def test_returns_descriptor_for_pipeline_enabled_glitchling(self) -> None:
        glitchling = MockGlitchling("Enabled", level=1, order=1, seed=42)
        descriptor = build_pipeline_descriptor(
            glitchling,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )

        assert descriptor is not None
        assert descriptor.name == "Enabled"
        assert descriptor.seed == 42
        assert descriptor.operation.get("type") == "mock_enabled"

    def test_returns_none_for_non_pipeline_glitchling(self) -> None:
        glitchling = MockGlitchling("Disabled", level=1, order=1, has_pipeline=False)
        descriptor = build_pipeline_descriptor(
            glitchling,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )

        assert descriptor is None

    def test_derives_seed_when_not_explicit(self) -> None:
        glitchling = MockGlitchling("DerivedSeed", level=1, order=1, seed=None, gaggle_index=5)
        descriptor = build_pipeline_descriptor(
            glitchling,
            master_seed=12345,
            derive_seed_fn=mock_derive_seed,
        )

        assert descriptor is not None
        expected_seed = mock_derive_seed(12345, "DerivedSeed", 5)
        assert descriptor.seed == expected_seed

    def test_raises_when_seed_cannot_be_determined(self) -> None:
        glitchling = MockGlitchling("NoSeed", level=1, order=1, seed=None)
        # No _gaggle_index attribute

        with pytest.raises(RuntimeError, match="missing deterministic seed"):
            build_pipeline_descriptor(
                glitchling,
                master_seed=100,
                derive_seed_fn=mock_derive_seed,
            )


# ---------------------------------------------------------------------------
# ExecutionStep and ExecutionPlan Tests
# ---------------------------------------------------------------------------


class TestExecutionStep:
    """Tests for ExecutionStep dataclass."""

    def test_pipeline_step(self) -> None:
        descriptors: tuple[PipelineDescriptor, ...] = (
            {"name": "Test", "seed": 1, "operation": {"type": "test"}},
        )
        step = ExecutionStep(descriptors, None)

        assert step.is_pipeline_step
        assert not step.is_fallback_step

    def test_fallback_step(self) -> None:
        glitchling = MockGlitchling("Fallback", level=1, order=1)
        step = ExecutionStep((), glitchling)

        assert not step.is_pipeline_step
        assert step.is_fallback_step


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_all_pipeline_flag(self) -> None:
        descriptors: tuple[PipelineDescriptor, ...] = (
            {"name": "A", "seed": 1, "operation": {"type": "a"}},
            {"name": "B", "seed": 2, "operation": {"type": "b"}},
        )
        step = ExecutionStep(descriptors, None)
        plan = ExecutionPlan((step,), all_pipeline=True)

        assert plan.all_pipeline
        assert plan.step_count == 1
        assert plan.pipeline_step_count == 1
        assert plan.fallback_step_count == 0

    def test_mixed_plan_counts(self) -> None:
        pipeline_step = ExecutionStep(
            ({"name": "P", "seed": 1, "operation": {"type": "p"}},),
            None,
        )
        fallback_step = ExecutionStep((), MockGlitchling("F", level=1, order=1, has_pipeline=False))
        plan = ExecutionPlan((pipeline_step, fallback_step), all_pipeline=False)

        assert not plan.all_pipeline
        assert plan.step_count == 2
        assert plan.pipeline_step_count == 1
        assert plan.fallback_step_count == 1


# ---------------------------------------------------------------------------
# build_execution_plan Tests
# ---------------------------------------------------------------------------


class TestBuildExecutionPlan:
    """Tests for build_execution_plan function."""

    def test_all_pipeline_glitchlings(self) -> None:
        glitchlings = [
            MockGlitchling("A", level=1, order=1, seed=1, gaggle_index=0),
            MockGlitchling("B", level=2, order=2, seed=2, gaggle_index=1),
        ]
        plan = build_execution_plan(
            glitchlings,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )

        assert plan.all_pipeline
        assert plan.step_count == 1
        assert len(plan.steps[0].descriptors) == 2

    def test_all_fallback_glitchlings(self) -> None:
        glitchlings = [
            MockGlitchling("A", level=1, order=1, has_pipeline=False, gaggle_index=0),
            MockGlitchling("B", level=2, order=2, has_pipeline=False, gaggle_index=1),
        ]
        plan = build_execution_plan(
            glitchlings,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )

        assert not plan.all_pipeline
        assert plan.step_count == 2
        assert plan.fallback_step_count == 2

    def test_mixed_pipeline_fallback_batching(self) -> None:
        """Test that consecutive pipeline glitchlings are batched together."""
        glitchlings = [
            MockGlitchling("P1", level=1, order=1, seed=1, gaggle_index=0),
            MockGlitchling("P2", level=2, order=2, seed=2, gaggle_index=1),
            MockGlitchling("F1", level=3, order=3, has_pipeline=False, gaggle_index=2),
            MockGlitchling("P3", level=4, order=4, seed=3, gaggle_index=3),
        ]
        plan = build_execution_plan(
            glitchlings,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )

        # Expected: [P1+P2 batch] -> [F1 fallback] -> [P3 batch]
        assert plan.step_count == 3
        assert plan.pipeline_step_count == 2
        assert plan.fallback_step_count == 1

        # First batch has P1 and P2
        assert len(plan.steps[0].descriptors) == 2
        # Second is fallback
        assert plan.steps[1].fallback_glitchling is not None
        # Third batch has P3
        assert len(plan.steps[2].descriptors) == 1

    def test_empty_apply_order(self) -> None:
        plan = build_execution_plan(
            [],
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )

        assert plan.all_pipeline
        assert plan.step_count == 0


# ---------------------------------------------------------------------------
# Plan Validation Helper Tests
# ---------------------------------------------------------------------------


class TestPlanValidationHelpers:
    """Tests for plan validation helper functions."""

    def test_validate_plan_coverage_complete(self) -> None:
        plan = [(0, 100), (1, 200), (2, 300)]
        missing = validate_plan_coverage(plan, 3)

        assert len(missing) == 0

    def test_validate_plan_coverage_incomplete(self) -> None:
        plan = [(0, 100), (2, 300)]  # Missing index 1
        missing = validate_plan_coverage(plan, 3)

        assert missing == {1}

    def test_extract_plan_ordering(self) -> None:
        plan = [(2, 300), (0, 100), (1, 200)]
        ordering = extract_plan_ordering(plan)

        assert ordering == [2, 0, 1]

    def test_extract_plan_seeds(self) -> None:
        plan = [(0, 100), (1, 200), (2, 300)]
        seeds = extract_plan_seeds(plan)

        assert seeds == {0: 100, 1: 200, 2: 300}


# ---------------------------------------------------------------------------
# build_plan_preview Tests
# ---------------------------------------------------------------------------


class _EnumLike:
    """Minimal enum stand-in exposing ``name``/integer value semantics."""

    def __init__(self, value: int, name: str) -> None:
        self.value = value
        self.name = name

    def __int__(self) -> int:
        return self.value


def _orchestration_plan(
    glitchlings: list[MockGlitchling], master_seed: int
) -> list[tuple[int, int]]:
    return [
        (index, mock_derive_seed(master_seed, glitchling.name, index))
        for index, glitchling in enumerate(glitchlings)
    ]


class TestBuildPlanPreview:
    """Tests for the serialisable execution-plan preview builder."""

    def test_preview_all_pipeline_single_batch(self) -> None:
        glitchlings = [
            MockGlitchling(
                "Alpha",
                level=_EnumLike(4, "CHARACTER"),
                order=_EnumLike(1, "EARLY"),
                seed=11,
                gaggle_index=0,
            ),
            MockGlitchling(
                "Beta",
                level=_EnumLike(1, "DOCUMENT"),
                order=_EnumLike(2, "NORMAL"),
                seed=22,
                gaggle_index=1,
            ),
        ]
        plan = build_execution_plan(
            glitchlings,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )
        preview = build_plan_preview(
            glitchlings,
            _orchestration_plan(glitchlings, 100),
            plan,
            master_seed=100,
        )

        assert preview["mode"] == "plan"
        assert preview["master_seed"] == 100
        assert preview["roster_count"] == 2
        assert preview["step_count"] == 1
        assert preview["pipeline_step_count"] == 1
        assert preview["fallback_step_count"] == 0
        assert preview["all_pipeline"] is True

        roster = preview["roster"]
        assert [entry["name"] for entry in roster] == ["Alpha", "Beta"]
        assert [entry["position"] for entry in roster] == [0, 1]
        assert [entry["source_index"] for entry in roster] == [0, 1]
        assert [entry["step"] for entry in roster] == [0, 0]
        assert {entry["execution"] for entry in roster} == {"pipeline"}
        assert [entry["operation_type"] for entry in roster] == [
            "mock_alpha",
            "mock_beta",
        ]
        first = roster[0]
        assert first["scope"] == "character"
        assert first["scope_value"] == 4
        assert first["order"] == "early"
        assert first["order_value"] == 1
        assert first["seed"] == mock_derive_seed(100, "Alpha", 0)

        steps = preview["steps"]
        assert steps == [
            {
                "step": 0,
                "execution": "pipeline",
                "size": 2,
                "members": ["Alpha", "Beta"],
                "operation_types": ["mock_alpha", "mock_beta"],
                "seeds": [
                    mock_derive_seed(100, "Alpha", 0),
                    mock_derive_seed(100, "Beta", 1),
                ],
            }
        ]

    def test_preview_mixed_pipeline_and_fallback_batches(self) -> None:
        glitchlings = [
            MockGlitchling("P1", level=1, order=1, seed=1, gaggle_index=0),
            MockGlitchling("F1", level=2, order=2, has_pipeline=False, gaggle_index=1),
            MockGlitchling("P2", level=3, order=3, seed=3, gaggle_index=2),
        ]
        plan = build_execution_plan(
            glitchlings,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )
        preview = build_plan_preview(
            glitchlings,
            _orchestration_plan(glitchlings, 100),
            plan,
            master_seed=100,
        )

        assert preview["step_count"] == 3
        assert preview["pipeline_step_count"] == 2
        assert preview["fallback_step_count"] == 1
        assert preview["all_pipeline"] is False

        roster = preview["roster"]
        assert [entry["execution"] for entry in roster] == [
            "pipeline",
            "fallback",
            "pipeline",
        ]
        assert [entry["operation_type"] for entry in roster] == [
            "mock_p1",
            None,
            "mock_p2",
        ]
        assert [entry["step"] for entry in roster] == [0, 1, 2]
        fallback_entry = roster[1]
        assert fallback_entry["name"] == "F1"
        assert fallback_entry["seed"] == mock_derive_seed(100, "F1", 1)

        assert [step["execution"] for step in preview["steps"]] == [
            "pipeline",
            "fallback",
            "pipeline",
        ]
        assert preview["steps"][1] == {
            "step": 1,
            "execution": "fallback",
            "size": 1,
            "members": ["F1"],
            "operation_types": None,
            "seeds": [mock_derive_seed(100, "F1", 1)],
        }

    def test_preview_all_fallback(self) -> None:
        glitchlings = [
            MockGlitchling("F1", level=1, order=1, has_pipeline=False, gaggle_index=0),
            MockGlitchling("F2", level=2, order=2, has_pipeline=False, gaggle_index=1),
        ]
        plan = build_execution_plan(
            glitchlings,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )
        preview = build_plan_preview(
            glitchlings,
            _orchestration_plan(glitchlings, 100),
            plan,
            master_seed=100,
        )

        assert preview["step_count"] == 2
        assert preview["fallback_step_count"] == 2
        assert preview["pipeline_step_count"] == 0
        assert all(step["size"] == 1 for step in preview["steps"])
        assert all(step["operation_types"] is None for step in preview["steps"])

    def test_preview_empty_roster(self) -> None:
        plan = build_execution_plan(
            [],
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )
        preview = build_plan_preview([], [], plan, master_seed=100)

        assert preview["roster_count"] == 0
        assert preview["step_count"] == 0
        assert preview["roster"] == []
        assert preview["steps"] == []
        assert preview["all_pipeline"] is True

    def test_preview_integer_levels_have_null_labels(self) -> None:
        glitchlings = [MockGlitchling("Plain", level=5, order=3, seed=1, gaggle_index=0)]
        plan = build_execution_plan(
            glitchlings,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )
        preview = build_plan_preview(
            glitchlings,
            _orchestration_plan(glitchlings, 100),
            plan,
            master_seed=100,
        )

        entry = preview["roster"][0]
        assert entry["scope"] is None
        assert entry["scope_value"] == 5
        assert entry["order"] is None
        assert entry["order_value"] == 3

    def test_preview_duplicate_names_keep_distinct_indexes_and_seeds(self) -> None:
        glitchlings = [
            MockGlitchling("Twin", level=1, order=1, seed=1, gaggle_index=0),
            MockGlitchling("Twin", level=1, order=1, seed=2, gaggle_index=1),
        ]
        plan = build_execution_plan(
            glitchlings,
            master_seed=100,
            derive_seed_fn=mock_derive_seed,
        )
        preview = build_plan_preview(
            glitchlings,
            _orchestration_plan(glitchlings, 100),
            plan,
            master_seed=100,
        )

        entries = preview["roster"]
        assert [entry["source_index"] for entry in entries] == [0, 1]
        assert [entry["position"] for entry in entries] == [0, 1]
        assert entries[0]["seed"] != entries[1]["seed"]
        assert entries[0]["seed"] == mock_derive_seed(100, "Twin", 0)
        assert entries[1]["seed"] == mock_derive_seed(100, "Twin", 1)
        assert preview["steps"][0]["members"] == ["Twin", "Twin"]
