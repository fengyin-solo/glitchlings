"""Tests for per-stage execution tracing.

Covers TraceConfig validation, the TraceRecorder observation helpers, the
Gaggle traced execution path, deterministic replay, failure prefixes, and
strict final-output parity with non-traced runs.
"""

from __future__ import annotations

import hashlib

import pytest
import yaml

from glitchlings import (
    Mim1c,
    Redactyl,
    Rushmore,
    Scannequin,
    TraceConfig,
    TraceRecorder,
    Typogre,
    Zeedub,
)
from glitchlings.constants import (
    DEFAULT_TRACE_PREVIEW_LIMIT,
    MAX_TRACE_PREVIEW_LIMIT,
)
from glitchlings.zoo.core import AttackOrder, AttackWave, Gaggle, Glitchling
from glitchlings.zoo.trace import (
    STATUS_FAILED,
    STATUS_OK,
    TRACE_PATH_FALLBACK,
    TRACE_PATH_PIPELINE,
    build_preview,
    count_changed_chars,
    text_sha256,
)
from glitchlings.zoo.validation import normalize_trace_format

UNICODE_TEXT = "café résumé naïve — Unicode 混合文本 αβγ 100%"


class _NoPipelineGlitchling(Glitchling):
    """Glitchling without Rust pipeline support, forcing Python fallback."""

    def __init__(
        self,
        name: str = "NoPipeline",
        seed: int | None = None,
        *,
        marker: str = "[fallback]",
        raises: RuntimeError | None = None,
    ) -> None:
        def _corrupt(
            text: str,
            *,
            rng: object,
            marker: str = marker,
            raises: object = raises,
        ) -> str:
            if raises is not None:
                raise raises
            return text + marker

        super().__init__(
            name=name,
            corruption_function=_corrupt,
            scope=AttackWave.DOCUMENT,
            order=AttackOrder.NORMAL,
            seed=seed,
            pipeline_operation=None,
            # Tracked as kwargs so Gaggle.clone() preserves the configuration.
            marker=marker,
            raises=raises,
        )

    def pipeline_operation(self) -> None:
        return None

    def clone(self, seed: int | None = None) -> "_NoPipelineGlitchling":
        return _NoPipelineGlitchling(
            name=self.name,
            seed=seed if seed is not None else self.seed,
            marker=self.kwargs["marker"],
            raises=self.kwargs["raises"],
        )


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


class TestTraceConfig:
    def test_default_preview_limit(self) -> None:
        assert TraceConfig().preview_limit == DEFAULT_TRACE_PREVIEW_LIMIT

    @pytest.mark.parametrize("value", [0, 1, 80, MAX_TRACE_PREVIEW_LIMIT])
    def test_accepts_values_within_hard_cap(self, value: int) -> None:
        assert TraceConfig(preview_limit=value).preview_limit == value

    @pytest.mark.parametrize("value", [-1, -100, MAX_TRACE_PREVIEW_LIMIT + 1, 10_000])
    def test_rejects_negative_and_oversized_before_execution(self, value: int) -> None:
        with pytest.raises(ValueError, match="trace preview limit"):
            TraceConfig(preview_limit=value)

    @pytest.mark.parametrize("value", [True, False, 1.5, "80", None])
    def test_rejects_non_integer_limits(self, value: object) -> None:
        with pytest.raises(TypeError, match="integer"):
            TraceConfig(preview_limit=value)  # type: ignore[arg-type]

    def test_trace_format_normalization(self) -> None:
        assert normalize_trace_format("json") == "json"
        assert normalize_trace_format("yaml") == "yaml"
        assert normalize_trace_format("YML") == "yaml"
        with pytest.raises(ValueError, match="unsupported trace format"):
            normalize_trace_format("toml")


# ---------------------------------------------------------------------------
# Recorder observation helpers
# ---------------------------------------------------------------------------


class TestRecorderHelpers:
    def test_text_sha256_matches_reference(self) -> None:
        text = "hash me"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert text_sha256(text) == expected

    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            ("abc", "abc", 0),
            ("abc", "axc", 1),
            ("abc", "ab", 1),
            ("ab", "abc", 1),
            ("", "abc", 3),
            ("abc", "", 3),
        ],
    )
    def test_count_changed_chars(self, before: str, after: str, expected: int) -> None:
        assert count_changed_chars(before, after) == expected

    def test_preview_reports_truncation(self) -> None:
        preview, truncated = build_preview("abcdef", 4)
        assert preview == "abcd"
        assert truncated is True
        preview, truncated = build_preview("ab", 4)
        assert preview == "ab"
        assert truncated is False

    def test_record_stage_captures_full_observation(self) -> None:
        recorder = TraceRecorder(master_seed=151)
        record = recorder.record_stage(
            name="Typogre",
            seed=12345,
            scope="character",
            order="early",
            execution_path=TRACE_PATH_PIPELINE,
            input_text="hello",
            output_text="heLlo",
        )
        assert record["index"] == 0
        assert record["status"] == STATUS_OK
        assert record["name"] == "Typogre"
        assert record["seed"] == 12345
        assert record["scope"] == "character"
        assert record["order"] == "early"
        assert record["execution_path"] == TRACE_PATH_PIPELINE
        assert record["input_length"] == 5
        assert record["output_length"] == 5
        assert record["input_hash"] == text_sha256("hello")
        assert record["output_hash"] == text_sha256("heLlo")
        assert record["changed_chars"] == 1
        assert record["input_preview"] == "hello"
        assert record["input_preview_truncated"] is False
        assert recorder.status == STATUS_OK
        assert recorder.completed_stages == 1

    def test_record_failure_keeps_prefix_and_marks_stage(self) -> None:
        recorder = TraceRecorder(master_seed=7)
        recorder.record_stage(
            name="First",
            seed=1,
            scope="document",
            order="normal",
            execution_path=TRACE_PATH_FALLBACK,
            input_text="in",
            output_text="out",
        )
        error = ValueError("boom")
        record = recorder.record_failure(
            name="Second",
            seed=2,
            scope="word",
            order="normal",
            execution_path=TRACE_PATH_PIPELINE,
            input_text="out",
            error=error,
        )
        assert record["status"] == STATUS_FAILED
        assert record["output_length"] is None
        assert record["output_hash"] is None
        assert record["changed_chars"] is None
        assert record["error"] == {"type": "ValueError", "message": "boom"}
        assert recorder.status == STATUS_FAILED
        assert recorder.completed_stages == 1
        payload = recorder.to_payload()
        assert payload["status"] == STATUS_FAILED
        assert payload["error"] == {
            "stage_index": 1,
            "stage_name": "Second",
            "execution_path": TRACE_PATH_PIPELINE,
            "completed_stages": 1,
            "type": "ValueError",
            "message": "boom",
        }

    def test_render_json_and_yaml_roundtrip(self) -> None:
        import json

        recorder = TraceRecorder(TraceConfig(preview_limit=10), master_seed=3)
        recorder.record_stage(
            name="Mim1c",
            seed=9,
            scope="character",
            order="last",
            execution_path=TRACE_PATH_PIPELINE,
            input_text=UNICODE_TEXT,
            output_text=UNICODE_TEXT + "!",
        )
        json_payload = json.loads(recorder.render("json"))
        assert json_payload["master_seed"] == 3
        assert json_payload["stages"][0]["input_preview"] == UNICODE_TEXT[:10]

        yaml_payload = yaml.safe_load(recorder.render("yaml"))
        assert yaml_payload["stages"][0]["name"] == "Mim1c"
        # Unicode must remain readable in both formats.
        assert UNICODE_TEXT[:4] in recorder.render("yaml")
        assert UNICODE_TEXT[:4] in recorder.render("json")


# ---------------------------------------------------------------------------
# Gaggle traced execution
# ---------------------------------------------------------------------------


def _make_gaggle(roster: list[Glitchling], seed: int = 42, **kwargs: object) -> Gaggle:
    return Gaggle([member.clone() for member in roster], seed=seed, **kwargs)


class TestTracedExecution:
    def test_single_stage_trace(self) -> None:
        gaggle = _make_gaggle([Typogre(rate=0.2)])
        recorder = gaggle.enable_trace()
        output = gaggle.corrupt("Hello world")

        assert isinstance(output, str)
        assert recorder.stages and len(recorder.stages) == 1
        stage = recorder.stages[0]
        assert stage["name"] == "Typogre"
        assert stage["execution_path"] == TRACE_PATH_PIPELINE
        assert stage["scope"] == "character"
        assert stage["order"] == "early"
        assert isinstance(stage["seed"], int)
        assert stage["input_hash"] == text_sha256("Hello world")
        assert stage["output_hash"] == text_sha256(output)
        assert stage["input_preview"] and stage["output_preview"]
        payload = recorder.to_payload()
        assert payload["master_seed"] == 42
        assert payload["stage_count"] == 1
        assert payload["status"] == STATUS_OK

    def test_multi_step_pipeline_splits_into_named_stages(self) -> None:
        roster = [Typogre(rate=0.3), Rushmore(rate=0.2), Mim1c(rate=0.2)]
        gaggle = _make_gaggle(roster)
        recorder = gaggle.enable_trace()
        gaggle.corrupt("One two three four five six")

        expected_names = [member.name for member in gaggle.apply_order]
        assert [stage["name"] for stage in recorder.stages] == expected_names
        assert len(recorder.stages) == len(roster)
        assert all(stage["execution_path"] == TRACE_PATH_PIPELINE for stage in recorder.stages)
        # Hashes chain across stages: output of one is input of the next.
        for previous, current in zip(recorder.stages, recorder.stages[1:]):
            assert previous["output_hash"] == current["input_hash"]
            assert previous["output_length"] == current["input_length"]

    def test_mixed_fallback_keeps_single_total_order(self) -> None:
        roster = [
            _NoPipelineGlitchling(),
            Typogre(rate=0.2),
            Rushmore(rate=0.2),
            _NoPipelineGlitchling(name="SecondFallback", marker="[tail]"),
        ]
        gaggle = _make_gaggle(roster)
        recorder = gaggle.enable_trace()
        output = gaggle.corrupt("Hello mixed order")

        expected_names = [member.name for member in gaggle.apply_order]
        assert [stage["name"] for stage in recorder.stages] == expected_names
        paths = [stage["execution_path"] for stage in recorder.stages]
        assert TRACE_PATH_FALLBACK in paths
        assert TRACE_PATH_PIPELINE in paths
        # DOCUMENT fallback stages sort before word/character pipeline stages.
        assert paths[0] == TRACE_PATH_FALLBACK
        assert isinstance(output, str)
        assert output.endswith("[tail]")

    def test_empty_text_records_zero_length_stages(self) -> None:
        gaggle = _make_gaggle([Typogre(rate=0.2), Zeedub(rate=0.2)])
        recorder = gaggle.enable_trace()
        output = gaggle.corrupt("")

        assert output == ""
        assert len(recorder.stages) == 2
        for stage in recorder.stages:
            assert stage["input_length"] == 0
            assert stage["output_length"] == 0
            assert stage["input_preview"] == ""
            assert stage["input_preview_truncated"] is False
            assert stage["input_hash"] == text_sha256("")

    def test_unicode_uses_code_point_lengths_and_readable_previews(self) -> None:
        gaggle = _make_gaggle([Typogre(rate=0.4), Scannequin(rate=0.3)])
        recorder = gaggle.enable_trace(TraceConfig(preview_limit=12))
        gaggle.corrupt(UNICODE_TEXT)

        first = recorder.stages[0]
        assert first["input_length"] == len(UNICODE_TEXT)
        assert first["input_preview"] == UNICODE_TEXT[:12]
        assert first["input_preview_truncated"] is True
        rendered = recorder.render("json")
        assert UNICODE_TEXT[:4] in rendered  # ensure_ascii=False

    def test_preview_hard_cap_is_enforced_by_config(self) -> None:
        with pytest.raises(ValueError, match="must not exceed"):
            TraceConfig(preview_limit=MAX_TRACE_PREVIEW_LIMIT + 1)
        config = TraceConfig(preview_limit=5)
        gaggle = _make_gaggle([Typogre(rate=0.0)])
        recorder = gaggle.enable_trace(config)
        gaggle.corrupt("abcdefghij")
        stage = recorder.stages[0]
        assert stage["input_preview"] == "abcde"
        assert stage["input_preview_truncated"] is True

    def test_deterministic_repeat_produces_identical_payloads(self) -> None:
        roster = [Typogre(rate=0.3), Rushmore(rate=0.2), Mim1c(rate=0.2)]
        text = "Deterministic replay requires identical payloads"

        first = _make_gaggle(roster, seed=77)
        first_recorder = first.enable_trace()
        first_output = first.corrupt(text)

        second = _make_gaggle(roster, seed=77)
        second_recorder = second.enable_trace()
        second_output = second.corrupt(text)

        assert first_output == second_output
        assert first_recorder.to_payload() == second_recorder.to_payload()

    @pytest.mark.parametrize(
        ("roster", "kwargs"),
        [
            ([Typogre(rate=0.4), Rushmore(rate=0.3)], {}),
            ([Typogre(rate=0.4), Scannequin(rate=0.3), Zeedub(rate=0.2)], {}),
            (
                [Rushmore(modes="duplicate", rate=1.0, seed=2)],
                {"exclude_patterns": [r"<br>"]},
            ),
            (
                [Typogre(rate=1.0, seed=5), Rushmore(modes="duplicate", rate=1.0, seed=5)],
                {"exclude_patterns": [r"STATIC"]},
            ),
            (
                [Typogre(rate=1.0, seed=5), Mim1c(rate=1.0)],
                {"include_only_patterns": [r"`[^`]+`"]},
            ),
            (
                [
                    Typogre(rate=0.5, exclude_patterns=[r"<a>"]),
                    Typogre(rate=0.5, exclude_patterns=[r"<b>"]),
                ],
                {},
            ),
            ([_NoPipelineGlitchling(), Typogre(rate=0.3), Rushmore(rate=0.2)], {}),
        ],
    )
    def test_traced_output_equals_normal_run(
        self,
        roster: list[Glitchling],
        kwargs: dict[str, object],
    ) -> None:
        text = "alpha <br> beta STATIC gamma `code block` delta probe 100%"
        normal_gaggle = _make_gaggle(roster, seed=11, **kwargs)
        traced_gaggle = _make_gaggle(roster, seed=11, **kwargs)
        recorder = traced_gaggle.enable_trace()

        normal_output = normal_gaggle.corrupt(text)
        traced_output = traced_gaggle.corrupt(text)

        assert traced_output == normal_output
        assert len(recorder.stages) == len(traced_gaggle.apply_order)
        assert recorder.stages[-1]["output_hash"] == text_sha256(traced_output)

    def test_fallback_failure_marks_stage_and_reraises_with_prefix(self) -> None:
        boom = RuntimeError("fallback exploded")
        roster = [
            _NoPipelineGlitchling(name="AaaPrefix"),
            _NoPipelineGlitchling(name="Boom", raises=boom),
            Typogre(rate=0.2),
        ]
        gaggle = _make_gaggle(roster)
        recorder = gaggle.enable_trace()

        with pytest.raises(RuntimeError, match="fallback exploded"):
            gaggle.corrupt("Hello failure")

        payload = recorder.to_payload()
        assert payload["status"] == STATUS_FAILED
        assert payload["error"] is not None
        assert payload["error"]["stage_name"] == "Boom"
        assert payload["error"]["completed_stages"] == 1
        assert payload["error"]["type"] == "RuntimeError"
        statuses = [stage["status"] for stage in recorder.stages]
        assert statuses == [STATUS_OK, STATUS_FAILED]
        # Stages after the failure never ran.
        assert len(recorder.stages) == 2

    def test_pipeline_failure_marks_stage_and_reraises(self) -> None:
        gaggle = _make_gaggle([Redactyl(rate=1.0)])
        recorder = gaggle.enable_trace()

        with pytest.raises(ValueError, match="redactable"):
            gaggle.corrupt("   ")

        payload = recorder.to_payload()
        assert payload["status"] == STATUS_FAILED
        failed = payload["stages"][-1]
        assert failed["name"] == "Redactyl"
        assert failed["status"] == STATUS_FAILED
        assert failed["execution_path"] == TRACE_PATH_PIPELINE
        assert failed["error"]["type"] == "ValueError"
        assert failed["output_preview"] is None

    def test_pipeline_failure_with_completed_prefix(self) -> None:
        roster = [_NoPipelineGlitchling(marker=" "), Redactyl(rate=1.0)]
        gaggle = _make_gaggle(roster)
        recorder = gaggle.enable_trace()

        with pytest.raises(ValueError, match="redactable"):
            gaggle.corrupt("   ")

        payload = recorder.to_payload()
        assert payload["error"] is not None
        assert payload["error"]["completed_stages"] == 1
        assert [stage["status"] for stage in payload["stages"]] == [
            STATUS_OK,
            STATUS_FAILED,
        ]

    def test_disable_trace_restores_default_execution(self) -> None:
        roster = [Typogre(rate=0.3), Rushmore(rate=0.2)]
        gaggle = _make_gaggle(roster)
        assert gaggle._trace_recorder is None

        recorder = gaggle.enable_trace()
        assert gaggle._trace_recorder is recorder
        gaggle.corrupt("Disable me")
        assert len(recorder.stages) == 2

        gaggle.disable_trace()
        assert gaggle._trace_recorder is None
        gaggle.corrupt("Disable me")
        # No new stages were recorded after disabling.
        assert len(recorder.stages) == 2
