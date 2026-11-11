"""CLI tests for the --trace family of options."""

from __future__ import annotations

import json

import pytest
import yaml

from glitchlings.constants import MAX_TRACE_PREVIEW_LIMIT
from glitchlings.main import build_parser, run_cli


def invoke(arguments: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(arguments)
    return run_cli(args, parser)


def _extract_json_trace(stdout: str) -> dict:
    marker = "{\n  "
    start = stdout.index(marker)
    return json.loads(stdout[start:])


def test_trace_stdout_keeps_corrupted_output_and_appends_json(capsys) -> None:
    exit_code = invoke(["--trace", "-g", "Typogre(rate=0.2)", "Hello world"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = _extract_json_trace(captured.out)
    assert payload["status"] == "ok"
    assert payload["stage_count"] == 1
    stage = payload["stages"][0]
    assert stage["name"] == "Typogre"
    assert stage["execution_path"] == "rust_pipeline"
    assert {"seed", "scope", "order", "input_hash", "output_hash"}.issubset(stage)
    # The corrupted text remains the first thing printed on stdout.
    marker = "{\n  "
    corrupted_prefix = captured.out[: captured.out.index(marker)]
    assert corrupted_prefix.endswith("\n")
    assert corrupted_prefix.strip()


def test_trace_yaml_and_yml_alias(capsys) -> None:
    invoke(["--trace", "--trace-format", "yaml", "-g", "Typogre(rate=0.1)", "Hello"])
    yaml_output = capsys.readouterr().out
    yaml_payload = yaml.safe_load(yaml_output[yaml_output.index("status:") :])
    assert yaml_payload["status"] == "ok"
    assert yaml_payload["stages"][0]["name"] == "Typogre"

    invoke(["--trace", "--trace-format", "yml", "-g", "Typogre(rate=0.1)", "Hello"])
    aliased_output = capsys.readouterr().out
    aliased = yaml.safe_load(aliased_output[aliased_output.index("status:") :])
    assert aliased["status"] == "ok"


def test_trace_output_file_keeps_stdout_clean(tmp_path, capsys) -> None:
    trace_path = tmp_path / "trace.json"
    exit_code = invoke(
        [
            "--trace",
            "--trace-output",
            str(trace_path),
            "-g",
            "Typogre(rate=0.2)",
            "Hello world",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "{" not in captured.out  # trace never leaks to stdout
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["stage_count"] == 1
    assert payload["stages"][0]["input_preview"] == "Hello world"
    assert captured.out.strip()  # corrupted text still printed to stdout


def test_trace_limit_truncates_previews(tmp_path) -> None:
    trace_path = tmp_path / "trace.json"
    invoke(
        [
            "--trace",
            "--trace-limit",
            "5",
            "--trace-output",
            str(trace_path),
            "-g",
            "Typogre(rate=0.0)",
            "abcdefghij",
        ]
    )
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    stage = payload["stages"][0]
    assert payload["preview_limit"] == 5
    assert stage["input_preview"] == "abcde"
    assert stage["input_preview_truncated"] is True


@pytest.mark.parametrize("bad_limit", [-1, -100, MAX_TRACE_PREVIEW_LIMIT + 1])
def test_trace_rejects_negative_and_oversized_limits(bad_limit: int, capsys) -> None:
    with pytest.raises(SystemExit):
        invoke(["--trace", "--trace-limit", str(bad_limit), "Hello"])
    captured = capsys.readouterr()
    assert "trace preview limit" in captured.err


def test_trace_limit_non_integer_rejected(capsys) -> None:
    with pytest.raises(SystemExit):
        invoke(["--trace", "--trace-limit", "abc", "Hello"])
    assert "invalid int value" in capsys.readouterr().err


def test_trace_suboptions_require_trace_flag(capsys) -> None:
    with pytest.raises(SystemExit):
        invoke(["--trace-format", "json", "Hello"])
    assert "require --trace" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        invoke(["--trace-limit", "10", "Hello"])
    assert "require --trace" in capsys.readouterr().err


def test_trace_conflicts_with_diff_and_attack(capsys) -> None:
    with pytest.raises(SystemExit):
        invoke(["--trace", "--diff", "Hello"])
    assert "--trace cannot be combined with --diff" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        invoke(["--trace", "--attack", "Hello"])
    assert "--trace cannot be combined with --attack or --report" in capsys.readouterr().err


def test_trace_invalid_format_choice(capsys) -> None:
    with pytest.raises(SystemExit):
        invoke(["--trace", "--trace-format", "toml", "Hello"])
    assert "invalid choice" in capsys.readouterr().err


def test_trace_output_must_differ_from_output_file(tmp_path, capsys) -> None:
    shared = tmp_path / "shared.txt"
    with pytest.raises(SystemExit):
        invoke(
            [
                "--trace",
                "--output-file",
                str(shared),
                "--trace-output",
                str(shared),
                "Hello",
            ]
        )
    assert "--trace-output must differ from --output-file" in capsys.readouterr().err


def test_trace_failure_writes_partial_trace_and_reraises(tmp_path) -> None:
    trace_path = tmp_path / "failed.json"
    parser = build_parser()
    args = parser.parse_args(
        [
            "--trace",
            "--trace-output",
            str(trace_path),
            "-g",
            "Redactyl(rate=1.0)",
            "   ",
        ]
    )

    with pytest.raises(ValueError, match="redactable"):
        run_cli(args, parser)

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] is not None
    assert payload["error"]["stage_name"] == "Redactyl"
    assert payload["error"]["completed_stages"] == 0
    assert payload["stages"][-1]["status"] == "failed"
    assert payload["stages"][-1]["error"]["type"] == "ValueError"


def test_trace_does_not_change_final_output(tmp_path, capsys) -> None:
    trace_path = tmp_path / "trace.json"
    parser = build_parser()

    plain_args = parser.parse_args(["-g", "Typogre(rate=0.2)", "Hello world"])
    run_cli(plain_args, parser)
    plain_output = capsys.readouterr().out

    traced_args = parser.parse_args(
        [
            "--trace",
            "--trace-output",
            str(trace_path),
            "-g",
            "Typogre(rate=0.2)",
            "Hello world",
        ]
    )
    run_cli(traced_args, parser)
    traced_output = capsys.readouterr().out

    assert traced_output == plain_output


def test_trace_is_deterministic_across_runs(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    arguments = [
        "-g",
        "Typogre(rate=0.2)",
        "-g",
        "Rushmore(rate=0.1)",
        "Deterministic trace replay",
    ]
    invoke(["--trace", "--trace-output", str(first), *arguments])
    invoke(["--trace", "--trace-output", str(second), *arguments])
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
