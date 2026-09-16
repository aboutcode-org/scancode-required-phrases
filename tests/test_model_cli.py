# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

from click.testing import CliRunner
import pytest

from licensedcode.models import Rule

from scancode_required_phrases import model_cli
from scancode_required_phrases import review
from scancode_required_phrases import review_ui
from scancode_required_phrases.inference import PhrasePrediction
from scancode_required_phrases.inference import PredictionResult


TEXT = (
    "Permission is granted under the MIT License to use copy modify merge publish "
    "distribute sublicense and sell copies of this software without restriction"
)


def make_rule(identifier="mit_test.RULE", text=TEXT):
    return Rule(
        identifier=identifier,
        license_expression="mit",
        text=text,
        is_license_notice=True,
        relevance=100,
    )


def write_rule(tmp_path, identifier="mit_test.RULE", text=TEXT):
    rule = make_rule(identifier, text)
    rule.dump(str(tmp_path))
    return (tmp_path / identifier).resolve()


class FakePredictor:
    def __init__(self, predictions=None, truncated=False):
        self.predictions = predictions or [
            PhrasePrediction("MIT License", 5, 6, 0.9),
        ]
        self.truncated = truncated
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        return PredictionResult(
            words=tuple(text.split()),
            phrases=tuple(self.predictions),
            truncated=self.truncated,
        )


def use_fake_model(monkeypatch, predictor=None):
    predictor = predictor or FakePredictor()
    loads = []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return predictor

    monkeypatch.setattr(model_cli, "load_predictor", load)
    return predictor, loads


def use_tty(monkeypatch):
    monkeypatch.setattr(model_cli, "stdin_is_tty", lambda: True)


def invoke_rule(runner, rule_path, *options, input=None):
    return runner.invoke(
        model_cli.add_model_required_phrases,
        ["--rule", str(rule_path), "--model", "unused", *options],
        input=input,
    )


@pytest.mark.parametrize(
    "options,message",
    [
        ([], "exactly one"),
        (["--all", "--rule", "x.RULE", "--model", "unused"], "exactly one"),
        (
            ["--all", "--model", "unused", "--predict-only", "--batch"],
            "mutually exclusive",
        ),
        (["--all", "--model", "unused", "--json", "-"], "requires --predict-only"),
        (
            ["--all", "--model", "unused", "--predict-only", "--session", "x"],
            "cannot be used",
        ),
        (["--all", "--model", "unused", "--batch"], "requires --auto-score"),
        (
            [
                "--all",
                "--model",
                "unused",
                "--batch",
                "--auto-score",
                "0.6",
                "--review-score",
                "0.8",
            ],
            "cannot exceed",
        ),
        (["--all", "--model", "unused", "--yes"], "require --batch"),
    ],
)
def test_command_rejects_invalid_option_combinations(options, message):
    result = CliRunner().invoke(model_cli.add_model_required_phrases, options)

    assert result.exit_code != 0
    assert message in result.output


def test_model_resolution_uses_only_the_pinned_default():
    assert model_cli.resolve_model(None, None) == (
        model_cli.DEFAULT_MODEL,
        model_cli.DEFAULT_MODEL_REVISION,
    )
    assert model_cli.resolve_model("local-model", None) == ("local-model", None)
    assert model_cli.resolve_model("owner/model", "a" * 40) == (
        "owner/model",
        "a" * 40,
    )


def test_command_rejects_resume_with_new_run_options(tmp_path, monkeypatch):
    use_tty(monkeypatch)
    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        ["--resume", str(tmp_path / "session.jsonl"), "--all"],
    )

    assert result.exit_code != 0
    assert "cannot be used with new-run options" in result.output


def test_command_rejects_non_tty_before_loading_model(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    monkeypatch.setattr(model_cli, "stdin_is_tty", lambda: False)
    monkeypatch.setattr(
        model_cli,
        "load_prediction_rule",
        lambda *args, **kwargs: pytest.fail("rule selection must not run"),
    )
    monkeypatch.setattr(
        model_cli,
        "load_predictor",
        lambda *args, **kwargs: pytest.fail("model must not load"),
    )

    result = invoke_rule(CliRunner(), rule_path)

    assert result.exit_code != 0
    assert "interactive terminal" in result.output


def test_command_rejects_existing_session_before_loading_model(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    session_path.write_text("keep", encoding="utf-8")
    use_tty(monkeypatch)
    monkeypatch.setattr(
        model_cli,
        "load_predictor",
        lambda *args, **kwargs: pytest.fail("model must not load"),
    )

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
    )

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert session_path.read_text(encoding="utf-8") == "keep"


def test_predict_only_json_stdout_is_clean(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    _predictor, loads = use_fake_model(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--predict-only",
        "--json",
        "-",
    )

    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert rows == [
        {
            "path": str(rule_path),
            "identifier": "mit_test.RULE",
            "license_expression": "mit",
            "phrase": "MIT License",
            "score": 0.9,
            "start_word": 5,
            "end_word": 6,
            "truncated": False,
            "validation_issue": None,
        }
    ]
    assert "Selecting rules..." in result.stderr
    assert "Loading model..." in result.stderr
    assert "Model loaded." in result.stderr
    assert len(loads) == 1
    assert "{{" not in Rule.from_file(str(rule_path)).text


def test_predict_only_json_stdout_is_empty_array_without_eligible_rules(monkeypatch):
    monkeypatch.setattr(model_cli, "select_installed_prediction_rules", lambda expression: [])
    monkeypatch.setattr(model_cli, "rules_data_dir", str(Path.cwd()))
    monkeypatch.setattr(
        model_cli,
        "load_predictor",
        lambda *args, **kwargs: pytest.fail("model must not load"),
    )

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        ["--all", "--model", "unused", "--predict-only", "--json", "-"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    assert "No eligible rules found" in result.stderr


def test_new_run_uses_pinned_default_model(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    _predictor, loads = use_fake_model(monkeypatch)

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        ["--rule", str(rule_path), "--predict-only"],
    )

    assert result.exit_code == 0, result.output
    assert loads == [
        (
            (model_cli.DEFAULT_MODEL,),
            {"hf_token": None, "revision": model_cli.DEFAULT_MODEL_REVISION},
        )
    ]


def test_predict_only_prints_validation_status(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    use_fake_model(
        monkeypatch,
        FakePredictor([PhrasePrediction("is", 1, 1, 0.99)], truncated=True),
    )

    result = invoke_rule(CliRunner(), rule_path, "--predict-only")

    assert result.exit_code == 0, result.output
    assert "mit_test.RULE" in result.output
    assert "99.0%" in result.output
    assert "rejected" in result.output
    assert "truncated" in result.output


def test_predict_only_writes_json_file(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    output_path = tmp_path / "output" / "predictions.json"
    use_fake_model(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--predict-only",
        "--json",
        str(output_path),
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output_path.read_text(encoding="utf-8"))[0]["phrase"] == "MIT License"


def test_interactive_dry_run_shows_review_and_writes_nothing(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    before = rule_path.read_bytes()
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        "--dry-run",
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    assert session_path.is_file()
    for text in (
        "[1/1] mit_test.RULE",
        "MIT License  90.0%",
        "expression: mit",
        "context:",
        "{{MIT License}}",
        "Rules: scanned 1 | ready 1 | deferred 0 | unchanged 0 | written 0",
        "Dry run: no rules were written",
    ):
        assert text in result.output


def test_verbose_review_shows_full_path(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        "--dry-run",
        "--verbose",
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert f"path: {rule_path}" in result.output
    assert "expression: mit" in result.output
    assert f"Session: {session_path}" in result.output


def test_interactive_save_exits_without_writing(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    before = rule_path.read_bytes()
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="y\ns\n",
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    assert "Session saved:" in result.output


def test_interactive_apply_writes_the_exact_rule_once(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)
    writes = []
    real_write = model_cli.write_rule_updates

    def write(*args, **kwargs):
        writes.append(args[0])
        return real_write(*args, **kwargs)

    monkeypatch.setattr(model_cli, "write_rule_updates", write)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="y\na\n",
    )

    assert result.exit_code == 0, result.output
    saved = Rule.from_file(str(rule_path))
    assert "{{MIT License}}" in saved.text
    assert saved.source == "ml_model"
    assert writes == [session_path]
    assert "written 1" in result.output


def test_interactive_reject_writes_nothing(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    before = rule_path.read_bytes()
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    _metadata, records = review.read_session(session_path)
    assert records[0]["predictions"][0]["decision"] == review.REJECTED


def test_interactive_edit_preserves_prediction(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        "--dry-run",
        input="e\nMIT License to use\n",
    )

    assert result.exit_code == 0, result.output
    _metadata, records = review.read_session(session_path)
    prediction = records[0]["predictions"][0]
    assert prediction["predicted_text"] == "MIT License"
    assert prediction["text"] == "MIT License to use"
    assert prediction["decision"] == review.APPROVED
    assert prediction["decision_source"] == "human"
    counts = review_ui.session_summary(_metadata, records)
    assert counts["approved"] == 0
    assert counts["edited"] == 1


def test_interactive_help_then_rejects(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="?\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Approve, reject, edit, skip for later" in result.output


def test_skip_leaves_pending_and_resume_does_not_load_model(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)
    runner = CliRunner()

    skipped = invoke_rule(
        runner,
        rule_path,
        "--session",
        str(session_path),
        input="s\n",
    )
    assert skipped.exit_code == 0, skipped.output
    assert "Resume with:" in skipped.output
    _metadata, records = review.read_session(session_path)
    assert records[0]["predictions"][0]["decision"] == review.PENDING

    monkeypatch.setattr(
        model_cli,
        "load_predictor",
        lambda *args, **kwargs: pytest.fail("resume must not load a model"),
    )
    resumed = runner.invoke(
        model_cli.add_model_required_phrases,
        ["--resume", str(session_path), "--dry-run"],
        input="y\n",
    )

    assert resumed.exit_code == 0, resumed.output
    assert "Dry run: no rules were written" in resumed.output


def test_skip_moves_to_next_prediction_without_reappearing(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    predictor = FakePredictor(
        [
            PhrasePrediction("MIT License", 5, 6, 0.9),
            PhrasePrediction("without restriction", 21, 22, 0.7),
        ]
    )
    use_fake_model(monkeypatch, predictor)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="s\nn\n",
    )

    assert result.exit_code == 0, result.output
    _metadata, records = review.read_session(session_path)
    first, second = records[0]["predictions"]
    assert first["decision"] == review.PENDING
    assert second["decision"] == review.REJECTED
    assert result.output.count("[1/2]") == 1
    assert result.output.count("[2/2]") == 1


def test_interactive_refuses_overlapping_approval(tmp_path, monkeypatch):
    rule_path = write_rule(
        tmp_path,
        text="binary Redistribution clause applies to these software copies",
    )
    session_path = tmp_path / "session.jsonl"
    predictor = FakePredictor(
        [
            PhrasePrediction("binary Redistribution clause", 0, 2, 0.9),
            PhrasePrediction("Redistribution clause", 1, 2, 0.8),
        ]
    )
    use_fake_model(monkeypatch, predictor)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="y\ny\nn\nd\n",
    )

    assert result.exit_code == 0, result.output
    assert "cannot be approved with the current decisions" in result.output
    _metadata, records = review.read_session(session_path)
    first, second = records[0]["predictions"]
    assert first["decision"] == review.APPROVED
    assert second["decision"] == review.REJECTED


def test_quit_prints_exact_resume_command(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session file.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="q\n",
    )

    assert result.exit_code == 0, result.output
    assert result.output.count(model_cli.resume_command(session_path)) == 1


def test_batch_classifies_exact_score_boundaries_and_blocks_mixed_write(
    tmp_path,
    monkeypatch,
):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "batch.jsonl"
    predictions = [
        PhrasePrediction("MIT License", 5, 6, 0.8),
        PhrasePrediction("without restriction", 21, 22, 0.5),
        PhrasePrediction("Permission is granted", 0, 2, 0.49),
    ]
    use_fake_model(monkeypatch, FakePredictor(predictions))
    before = rule_path.read_bytes()

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--batch",
        "--auto-score",
        "0.8",
        "--review-score",
        "0.5",
        "--yes",
        "--session",
        str(session_path),
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    metadata, records = review.read_session(session_path)
    first, second, third = records[0]["predictions"]
    assert (first["decision"], first["decision_source"]) == (review.APPROVED, "score")
    assert second["decision"] == review.PENDING
    assert review.prediction_needs_review(metadata, second)
    assert review.prediction_is_below_threshold(metadata, third)
    assert "Resume with:" in result.output


def test_batch_applies_ready_rule_and_defers_pending_rule(tmp_path, monkeypatch):
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    first_path = write_rule(rules_directory, "first.RULE")
    second_path = write_rule(rules_directory, "second.RULE")
    session_path = tmp_path / "batch.jsonl"

    class PerRulePredictor:
        def __init__(self):
            self.calls = 0

        def predict(self, text):
            self.calls += 1
            predictions = [PhrasePrediction("MIT License", 5, 6, 0.9)]
            if self.calls == 2:
                predictions.append(PhrasePrediction("without restriction", 21, 22, 0.7))
            return PredictionResult(
                words=tuple(text.split()),
                phrases=tuple(predictions),
                truncated=False,
            )

    use_fake_model(monkeypatch, PerRulePredictor())

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        [
            "--rules-dir",
            str(rules_directory),
            "--batch",
            "--auto-score",
            "0.8",
            "--review-score",
            "0.5",
            "--yes",
            "--session",
            str(session_path),
            "--model",
            "unused",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "{{MIT License}}" in Rule.from_file(str(first_path)).text
    assert "{{" not in Rule.from_file(str(second_path)).text
    _metadata, records = review.read_session(session_path)
    assert records[0]["applied_hash"] is not None
    assert records[1]["applied_hash"] is None
    assert "deferred 1" in result.output
    assert "Resume with:" in result.output


def test_batch_without_yes_saves_without_writing(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "batch.jsonl"
    before = rule_path.read_bytes()
    use_fake_model(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--batch",
        "--auto-score",
        "0.8",
        "--review-score",
        "0.5",
        "--session",
        str(session_path),
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    assert "Session saved:" in result.output


def test_batch_dry_run_overrides_yes(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "batch.jsonl"
    before = rule_path.read_bytes()
    use_fake_model(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--batch",
        "--auto-score",
        "0.8",
        "--review-score",
        "0.5",
        "--yes",
        "--dry-run",
        "--session",
        str(session_path),
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    assert "Dry run: no rules were written" in result.output


def test_batch_yes_applies_when_no_review_is_pending(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "batch.jsonl"
    use_fake_model(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--batch",
        "--auto-score",
        "0.8",
        "--review-score",
        "0.5",
        "--yes",
        "--session",
        str(session_path),
    )

    assert result.exit_code == 0, result.output
    assert "{{MIT License}}" in Rule.from_file(str(rule_path)).text


def test_batch_never_score_approves_truncated_rule(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "batch.jsonl"
    before = rule_path.read_bytes()
    use_fake_model(monkeypatch, FakePredictor(truncated=True))

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--batch",
        "--auto-score",
        "0.8",
        "--review-score",
        "0.5",
        "--yes",
        "--session",
        str(session_path),
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    metadata, records = review.read_session(session_path)
    assert records[0]["predictions"][0]["decision"] == review.PENDING
    assert metadata["truncated_rules"] == 1


def test_batch_score_does_not_override_candidate_validation(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "batch.jsonl"
    before = rule_path.read_bytes()
    use_fake_model(monkeypatch, FakePredictor([PhrasePrediction("is", 1, 1, 1.0)]))

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--batch",
        "--auto-score",
        "0.8",
        "--review-score",
        "0.5",
        "--yes",
        "--session",
        str(session_path),
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    _metadata, records = review.read_session(session_path)
    prediction = records[0]["predictions"][0]
    assert prediction["validation_issue"] == "rejected"
    assert prediction["decision"] == review.PENDING


def test_directory_limit_predicts_only_selected_rules(tmp_path, monkeypatch):
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    write_rule(rules_directory, "b.RULE")
    write_rule(rules_directory, "a.RULE")
    predictor, loads = use_fake_model(monkeypatch)

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        [
            "--rules-dir",
            str(rules_directory),
            "--limit",
            "1",
            "--model",
            "unused",
            "--predict-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(loads) == 1
    assert predictor.calls == [TEXT]
    assert "a.RULE" in result.output
    assert "b.RULE" not in result.output


def test_all_target_reuses_installed_selection(tmp_path, monkeypatch):
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    rule_path = write_rule(rules_directory)
    rule = Rule.from_file(str(rule_path))
    selected = [(rule_path, rule)]
    calls = []
    monkeypatch.setattr(
        model_cli,
        "select_installed_prediction_rules",
        lambda expression: calls.append(expression) or selected,
    )
    monkeypatch.setattr(model_cli, "rules_data_dir", str(rules_directory))
    use_fake_model(monkeypatch)

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        [
            "--all",
            "--license-expression",
            "mit",
            "--model",
            "unused",
            "--predict-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls == ["mit"]


def test_no_color_removes_terminal_escape_sequences(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        "--dry-run",
        "--no-color",
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "\x1b[" not in result.output


def test_command_help_is_single_workflow():
    result = CliRunner().invoke(model_cli.add_model_required_phrases, ["--help"])

    assert result.exit_code == 0
    assert "--rule" in result.output
    assert "--rules-dir" in result.output
    assert "--all" in result.output
    assert "--predict-only" in result.output
    assert "--batch" in result.output
    assert "--resume" in result.output
