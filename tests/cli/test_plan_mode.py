"""Tests for the CLI ``--plan`` execution-plan preview mode."""

from __future__ import annotations

import json

import pytest
import yaml

from glitchlings import Gaggle, Glitchling, summon
from glitchlings.conf import DEFAULT_ATTACK_SEED, build_gaggle, load_attack_config
from glitchlings.main import build_parser, run_cli
from glitchlings.zoo import Typogre
from glitchlings.zoo.core import AttackOrder, AttackWave


def run_plan(arguments: list[str], capsys: pytest.CaptureFixture[str]) -> dict:
    parser = build_parser()
    args = parser.parse_args(arguments)
    exit_code = run_cli(args, parser)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    return json.loads(captured.out)


class _ExplodingStdin:
    def isatty(self) -> bool:
        return True

    def read(self) -> str:
        raise AssertionError("plan mode must never read stdin")


def _expected_preview(specs: list[str], seed: int) -> dict:
    return summon(specs, seed=seed).preview_execution_plan()


def test_plan_mode_default_builtin_roster(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _ExplodingStdin())

    payload = run_plan(["--plan"], capsys)

    assert payload == _expected_preview(["typogre", "scannequin"], DEFAULT_ATTACK_SEED)
    assert payload["mode"] == "plan"
    assert payload["master_seed"] == DEFAULT_ATTACK_SEED
    assert payload["roster_count"] == 2
    # Scannequin is a DOCUMENT wave glitchling, so it sorts before CHARACTER Typogre.
    assert [entry["name"] for entry in payload["roster"]] == ["Scannequin", "Typogre"]
    assert [entry["position"] for entry in payload["roster"]] == [0, 1]
    assert [entry["source_index"] for entry in payload["roster"]] == [1, 0]


def test_plan_mode_inline_specs(capsys):
    payload = run_plan(
        ["--plan", "-g", "Typogre(rate=0.2)", "-g", "Zeedub"],
        capsys,
    )

    assert payload == _expected_preview(["Typogre(rate=0.2)", "Zeedub"], DEFAULT_ATTACK_SEED)
    names = [entry["name"] for entry in payload["roster"]]
    assert names == ["Typogre", "Zeedub"]
    assert [entry["operation_type"] for entry in payload["roster"]] == ["typo", "zwj"]
    assert all(entry["execution"] == "pipeline" for entry in payload["roster"])
    for entry in payload["roster"]:
        assert entry["scope"] == "character"
        assert entry["scope_value"] == int(AttackWave.CHARACTER)
        assert isinstance(entry["seed"], int)


def test_plan_mode_yaml_config(tmp_path, capsys):
    config_path = tmp_path / "attack.yaml"
    config_path.write_text(
        "seed: 12\nglitchlings:\n  - name: Typogre\n    rate: 0.02\n  - Zeedub\n",
        encoding="utf-8",
    )

    payload = run_plan(["--plan", "--config", str(config_path)], capsys)

    config = load_attack_config(config_path)
    assert payload == build_gaggle(config).preview_execution_plan()
    assert payload["master_seed"] == 12
    assert payload["roster_count"] == 2


def test_plan_mode_seed_override_changes_derived_seeds(capsys):
    default_payload = run_plan(["--plan", "-g", "Typogre"], capsys)
    override_payload = run_plan(["--plan", "-g", "Typogre", "--seed", "9"], capsys)

    assert default_payload["master_seed"] == DEFAULT_ATTACK_SEED
    assert override_payload["master_seed"] == 9
    assert (
        override_payload["roster"][0]["seed"]
        == Gaggle.derive_seed(9, "Typogre", override_payload["roster"][0]["source_index"])
    )
    assert (
        override_payload["roster"][0]["seed"] != default_payload["roster"][0]["seed"]
    )


def test_plan_mode_pipeline_merges_into_single_batch(capsys):
    payload = run_plan(
        ["--plan", "-g", "Typogre", "-g", "Scannequin", "-g", "Zeedub"],
        capsys,
    )

    assert payload["all_pipeline"] is True
    assert payload["step_count"] == 1
    assert payload["pipeline_step_count"] == 1
    assert payload["fallback_step_count"] == 0
    batch = payload["steps"][0]
    assert batch["execution"] == "pipeline"
    assert batch["size"] == 3
    assert len(batch["members"]) == 3
    assert len(batch["operation_types"]) == 3
    assert all(entry["step"] == 0 for entry in payload["roster"])


def test_plan_mode_fallback_mixing_splits_batches(monkeypatch, capsys):
    def _no_pipeline(self: Typogre) -> None:
        return None

    monkeypatch.setattr(Typogre, "pipeline_operation", _no_pipeline)

    payload = run_plan(["--plan", "-g", "Typogre", "-g", "Zeedub"], capsys)

    assert payload["all_pipeline"] is False
    assert payload["step_count"] == 2
    assert payload["pipeline_step_count"] == 1
    assert payload["fallback_step_count"] == 1

    roster = {entry["name"]: entry for entry in payload["roster"]}
    assert roster["Typogre"]["execution"] == "fallback"
    assert roster["Typogre"]["operation_type"] is None
    assert roster["Zeedub"]["execution"] == "pipeline"
    assert roster["Zeedub"]["operation_type"] == "zwj"

    steps = payload["steps"]
    assert [step["execution"] for step in steps] == ["fallback", "pipeline"]
    assert steps[0]["members"] == ["Typogre"]
    assert steps[1]["members"] == ["Zeedub"]
    # Each roster entry points at the step that actually executes it.
    for entry in payload["roster"]:
        assert entry["name"] in payload["steps"][entry["step"]]["members"]


def test_plan_mode_stable_sorting_matches_library(capsys):
    specs = ["Zeedub", "Scannequin", "Typogre"]
    payload = run_plan(["--plan", "-g", specs[0], "-g", specs[1], "-g", specs[2]], capsys)

    expected = summon(specs, seed=DEFAULT_ATTACK_SEED)
    expected_order = [
        (glitchling.name, int(glitchling.level), int(glitchling.order), glitchling.seed)
        for glitchling in expected.apply_order
    ]
    actual_order = [
        (entry["name"], entry["scope_value"], entry["order_value"], entry["seed"])
        for entry in payload["roster"]
    ]
    assert actual_order == expected_order
    assert payload == expected.preview_execution_plan()


def test_plan_mode_duplicate_roster_keeps_both_members(capsys):
    payload = run_plan(["--plan", "-g", "Typogre", "-g", "Typogre"], capsys)

    assert payload == _expected_preview(["Typogre", "Typogre"], DEFAULT_ATTACK_SEED)
    entries = payload["roster"]
    assert [entry["name"] for entry in entries] == ["Typogre", "Typogre"]
    assert [entry["source_index"] for entry in entries] == [0, 1]
    assert [entry["position"] for entry in entries] == [0, 1]
    assert entries[0]["seed"] != entries[1]["seed"]


def test_plan_mode_never_reads_stdin_or_runs_corruption(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _ExplodingStdin())

    def _explode(*args: object, **kwargs: object) -> str:
        raise AssertionError("plan mode must not execute corruption or pipelines")

    monkeypatch.setattr(Glitchling, "_execute_corruption", _explode)
    monkeypatch.setattr(
        "glitchlings.zoo.core_execution.compose_operations_rust",
        _explode,
    )
    monkeypatch.setattr(
        "glitchlings.internal.rust_ffi.RustPipeline.run",
        _explode,
    )

    payload = run_plan(["--plan", "-g", "Typogre", "-g", "Scannequin"], capsys)
    assert payload["roster_count"] == 2


def test_plan_mode_yaml_output(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _ExplodingStdin())

    parser = build_parser()
    args = parser.parse_args(["--plan", "--format", "yaml"])
    exit_code = run_cli(args, parser)
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = yaml.safe_load(captured.out)
    assert payload == _expected_preview(["typogre", "scannequin"], DEFAULT_ATTACK_SEED)


def test_plan_mode_yml_alias(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _ExplodingStdin())

    parser = build_parser()
    args = parser.parse_args(["--plan", "--format", "yml"])
    exit_code = run_cli(args, parser)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert yaml.safe_load(captured.out)["mode"] == "plan"


def test_plan_mode_writes_output_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _ExplodingStdin())
    output_path = tmp_path / "plan.json"

    parser = build_parser()
    args = parser.parse_args(["--plan", "--output-file", str(output_path)])
    exit_code = run_cli(args, parser)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == ""
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload == _expected_preview(["typogre", "scannequin"], DEFAULT_ATTACK_SEED)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--plan", "--attack", "--sample"], "--attack or --report"),
        (["--plan", "--report", "--sample"], "--attack or --report"),
        (["--plan", "--diff"], "--diff"),
        (["--plan", "--input-file", "input.txt"], "--input-file"),
        (["--plan", "--sample"], "--sample"),
        (["--plan", "--tokenizer", "cl100k_base"], "--tokenizer"),
        (["--plan", "positional", "text"], "does not accept input text"),
    ],
)
def test_plan_mode_rejects_conflicting_arguments(
    monkeypatch, capsys, arguments, message
):
    monkeypatch.setattr("sys.stdin", _ExplodingStdin())

    parser = build_parser()
    args = parser.parse_args(arguments)
    with pytest.raises(SystemExit):
        run_cli(args, parser)

    captured = capsys.readouterr()
    assert message in captured.err


def test_plan_mode_bad_spec_error_matches_real_run(capsys):
    parser = build_parser()
    plan_args = parser.parse_args(["--plan", "-g", "NoSuchGlitchling"])
    with pytest.raises(SystemExit):
        run_cli(plan_args, parser)
    plan_error = capsys.readouterr().err

    parser = build_parser()
    run_args = parser.parse_args(["-g", "NoSuchGlitchling", "Hello there"])
    with pytest.raises(SystemExit):
        run_cli(run_args, parser)
    run_error = capsys.readouterr().err

    assert plan_error == run_error
    assert "NoSuchGlitchling" in plan_error


def test_plan_mode_config_error_matches_real_run(tmp_path, capsys):
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("glitchlings:\n  - Nope\n", encoding="utf-8")

    parser = build_parser()
    plan_args = parser.parse_args(["--plan", "--config", str(config_path)])
    with pytest.raises(SystemExit):
        run_cli(plan_args, parser)
    plan_error = capsys.readouterr().err

    parser = build_parser()
    run_args = parser.parse_args(["--config", str(config_path), "Hello there"])
    with pytest.raises(SystemExit):
        run_cli(run_args, parser)
    run_error = capsys.readouterr().err

    assert plan_error == run_error
    assert "Nope" in plan_error


def test_plan_mode_missing_config_file_matches_real_run(tmp_path, capsys):
    missing = tmp_path / "missing.yaml"

    parser = build_parser()
    plan_args = parser.parse_args(["--plan", "--config", str(missing)])
    with pytest.raises(SystemExit):
        run_cli(plan_args, parser)
    plan_error = capsys.readouterr().err

    parser = build_parser()
    run_args = parser.parse_args(["--config", str(missing), "Hello there"])
    with pytest.raises(SystemExit):
        run_cli(run_args, parser)
    run_error = capsys.readouterr().err

    assert plan_error == run_error
    assert str(missing) in plan_error


def test_plan_mode_rejects_config_combined_with_glitchling(tmp_path, capsys):
    config_path = tmp_path / "attack.yaml"
    config_path.write_text("glitchlings:\n  - Typogre\n", encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(
        ["--plan", "--config", str(config_path), "--glitchling", "Typogre"]
    )
    with pytest.raises(SystemExit):
        run_cli(args, parser)

    assert "Cannot combine --config with --glitchling" in capsys.readouterr().err


def test_gaggle_preview_missing_seed_raises_like_orchestration():
    def _noop(text: str, rng: object) -> str:
        return text

    glitchling = Glitchling("Custom", _noop, AttackWave.WORD, AttackOrder.NORMAL)

    with pytest.raises(ValueError, match="master seed"):
        Gaggle([glitchling], seed=None)  # type: ignore[arg-type]


def test_gaggle_preview_payload_is_json_and_yaml_serialisable(capsys):
    payload = run_plan(["--plan", "-g", "Typogre", "-g", "Scannequin"], capsys)

    assert json.loads(json.dumps(payload)) == payload
    assert yaml.safe_load(yaml.safe_dump(payload, sort_keys=False)) == payload
