# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import difflib
import json
import os
from pathlib import Path

import click
from click.testing import CliRunner
import pytest

from licensedcode.models import Rule

from scancode_required_phrases import model_cli
from scancode_required_phrases import review
from scancode_required_phrases import review_ui
from scancode_required_phrases.inference import PhrasePrediction
from scancode_required_phrases.inference import PredictionResult
from scancode_required_phrases.inference import words_from_text


TEXT = (
    "Permission is granted under the MIT License to use copy modify merge publish "
    "distribute sublicense and sell copies of this software without restriction"
)


def make_rule(identifier="mit_test.RULE", text=TEXT, license_expression="mit", **kwargs):
    return Rule(
        identifier=identifier,
        license_expression=license_expression,
        text=text,
        is_license_notice=True,
        relevance=100,
        **kwargs,
    )


def write_rule(
    tmp_path,
    identifier="mit_test.RULE",
    text=TEXT,
    license_expression="mit",
    **kwargs,
):
    rule = make_rule(identifier, text, license_expression, **kwargs)
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
            words=tuple(words_from_text(text)),
            phrases=tuple(self.predictions),
            truncated=self.truncated,
        )


def use_fake_model(monkeypatch, predictor=None):
    predictor = predictor or FakePredictor()
    loads = []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        before_model_load = kwargs.get("before_model_load")
        if before_model_load:
            before_model_load()
        return predictor

    monkeypatch.setattr(model_cli, "load_predictor", load)
    return predictor, loads


def use_tty(monkeypatch):
    monkeypatch.setattr(model_cli, "stdin_is_tty", lambda: True)


def invoke_rule(runner, rule_path, *options, input=None, color=False):
    return runner.invoke(
        model_cli.add_model_required_phrases,
        ["--rule", str(rule_path), "--model", "unused", *options],
        input=input,
        color=color,
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
    assert "Checking model files..." in result.stderr
    assert "Fetching model files..." not in result.stderr
    assert "Loading model into memory..." in result.stderr
    assert "Model ready." in result.stderr
    assert result.stderr.index("Checking model files...") < result.stderr.index(
        "Loading model into memory..."
    )
    assert result.stderr.index("Loading model into memory...") < result.stderr.index("Model ready.")
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


def test_predict_only_reports_empty_selection_without_loading_model(monkeypatch):
    monkeypatch.setattr(model_cli, "select_installed_prediction_rules", lambda expression: [])
    monkeypatch.setattr(model_cli, "rules_data_dir", str(Path.cwd()))
    monkeypatch.setattr(
        model_cli,
        "load_predictor",
        lambda *args, **kwargs: pytest.fail("model must not load"),
    )

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        ["--all", "--model", "unused", "--predict-only"],
    )

    assert result.exit_code == 0, result.output
    assert "Selected rules:          0" in result.output
    assert "Rules eligible:" not in result.output
    assert "No eligible rules found." in result.output


def test_new_run_uses_pinned_default_model(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    _predictor, loads = use_fake_model(monkeypatch)

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        ["--rule", str(rule_path), "--predict-only"],
    )

    assert result.exit_code == 0, result.output
    assert len(loads) == 1
    args, kwargs = loads[0]
    assert args == (model_cli.DEFAULT_MODEL,)
    assert kwargs["hf_token"] is None
    assert kwargs["revision"] == model_cli.DEFAULT_MODEL_REVISION
    assert callable(kwargs["before_model_load"])


def test_local_model_progress_does_not_report_fetching(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    model_directory = tmp_path / "model"
    model_directory.mkdir()
    use_fake_model(monkeypatch)

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        [
            "--rule",
            str(rule_path),
            "--model",
            str(model_directory),
            "--predict-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Checking model files..." in result.output
    assert "Fetching model files..." not in result.output
    assert result.output.index("Loading model into memory...") < result.output.index("Model ready.")


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
    assert "Blocked - does not meet ScanCode required-phrase checks" in result.output
    assert "Rule input was truncated" in result.output


def test_predict_only_groups_predictions_by_rule(tmp_path, monkeypatch):
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    write_rule(rules_directory, "first.RULE")
    write_rule(rules_directory, "second.RULE")
    use_fake_model(
        monkeypatch,
        FakePredictor(
            [
                PhrasePrediction("MIT License", 5, 6, 0.9),
                PhrasePrediction("without restriction", 21, 22, 0.7),
            ]
        ),
    )

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        [
            "--rules-dir",
            str(rules_directory),
            "--model",
            "unused",
            "--predict-only",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.count("Rule 1 of 2") == 1
    assert result.output.count("Rule 2 of 2") == 1
    assert result.output.count("Expression: mit") == 2
    assert result.output.count("Predicted required phrases:") == 2
    assert result.output.count("{{MIT License}}") == 2
    assert result.output.count("{{without restriction}}") == 2
    assert "Selected rules:" in result.output
    assert "Rules with predictions:" in result.output
    assert "Predictions:             4 ready" in result.output
    assert "Predictions found:" not in result.output


def test_predict_only_explains_each_validation_result(tmp_path, monkeypatch):
    text = "MIT License appears here and MIT License appears again with another phrase"
    rule_path = write_rule(tmp_path, text=text)
    use_fake_model(
        monkeypatch,
        FakePredictor(
            [
                PhrasePrediction("another phrase", 10, 11, 0.9),
                PhrasePrediction("MIT License", 0, 1, 0.8),
                PhrasePrediction("missing phrase", 0, 1, 0.7),
                PhrasePrediction("is", 0, 0, 0.6),
            ]
        ),
    )

    result = invoke_rule(CliRunner(), rule_path, "--predict-only")

    assert result.exit_code == 0, result.output
    assert "Validation: Ready for review" in result.output
    assert "Blocked - phrase appears more than once" in result.output
    assert "Blocked - exact phrase was not found in the rule text" in result.output
    assert "Blocked - does not meet ScanCode required-phrase checks" in result.output


def test_predict_only_blocks_an_ignorable_url(tmp_path, monkeypatch):
    phrase = "spdx org licenses CC BY NC SA 2 0"
    rule_path = write_rule(
        tmp_path,
        text="License details https://spdx.org/licenses/CC-BY-NC-SA-2.0.html apply here",
        ignorable_urls=["https://spdx.org/licenses/CC-BY-NC-SA-2.0.html"],
    )
    use_fake_model(monkeypatch, FakePredictor([PhrasePrediction(phrase, 3, 11, 1.0)]))

    result = invoke_rule(CliRunner(), rule_path, "--predict-only")

    assert result.exit_code == 0, result.output
    assert "1 blocked by validation" in result.output
    assert "Blocked - does not meet ScanCode required-phrase checks" in result.output
    assert "Ready for review" not in result.output


def test_predict_only_json_blocks_an_ignorable_url(tmp_path, monkeypatch):
    phrase = "spdx org licenses CC BY NC SA 2 0"
    rule_path = write_rule(
        tmp_path,
        text="License details https://spdx.org/licenses/CC-BY-NC-SA-2.0.html apply here",
        ignorable_urls=["https://spdx.org/licenses/CC-BY-NC-SA-2.0.html"],
    )
    use_fake_model(monkeypatch, FakePredictor([PhrasePrediction(phrase, 3, 11, 1.0)]))

    result = invoke_rule(CliRunner(), rule_path, "--predict-only", "--json", "-")

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["validation_issue"] == "rejected"


def test_interactive_does_not_review_an_ignorable_url(tmp_path, monkeypatch):
    phrase = "spdx org licenses CC BY NC SA 2 0"
    rule_path = write_rule(
        tmp_path,
        text="License details https://spdx.org/licenses/CC-BY-NC-SA-2.0.html apply here",
        ignorable_urls=["https://spdx.org/licenses/CC-BY-NC-SA-2.0.html"],
    )
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch, FakePredictor([PhrasePrediction(phrase, 3, 11, 1.0)]))
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        "--dry-run",
    )

    assert result.exit_code == 0, result.output
    assert "Prediction 1 of 1" not in result.output
    assert "Actions:" not in result.output
    _metadata, records = review.read_session(session_path)
    prediction = records[0]["predictions"][0]
    assert prediction["validation_issue"] == "rejected"
    assert prediction["decision"] == review.PENDING


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
    assert result.output.index("\nReview\n") < result.output.index("Prediction 1 of 1")
    assert "Apply updates to" not in result.output
    assert result.output.count("\nSummary\n") == 1
    assert "Automatically approved" not in result.output
    assert "Below review threshold" not in result.output
    assert "Changed and approved:" not in result.output
    assert "Rejected:" not in result.output
    assert "Blocked by validation:" not in result.output
    for text in (
        "Review",
        "Prediction 1 of 1",
        "Predicted required phrase:",
        "{{MIT License}}",
        "Model score: 90.0%",
        "Validation:  Ready for review",
        "Expression: mit",
        "Context:",
        "Proposed change:",
        "--- a/mit_test.RULE",
        "+++ b/mit_test.RULE",
        "Actions:\n  [y] approve   [n] reject   [e] change",
        "Ready rules:",
        "Written rules:",
        "Dry run: no rule files were written.",
    ):
        assert text in result.output


def test_review_color_is_limited_to_phrase_markers_and_warning(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)
    monkeypatch.setattr(model_cli, "stdout_is_tty", lambda: True)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(tmp_path / "session.jsonl"),
        "--dry-run",
        input="n\n",
        color=True,
    )

    assert result.exit_code == 0, result.output
    assert "\x1b[32m" in result.output
    score_line = next(
        line for line in result.output.splitlines() if line.startswith("Model score:")
    )
    assert "\x1b[" not in score_line
    context = result.output.split("\nContext:\n", 1)[1].split("\nProposed change:\n", 1)[0]
    assert "\x1b[" not in context


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
    assert f"Path:       {rule_path}" in result.output
    assert "Expression: mit" in result.output
    approved_line = next(
        line for line in result.output.splitlines() if line.startswith("Approved phrases:")
    )
    changed_line = next(
        line for line in result.output.splitlines() if line.startswith("Changed and approved:")
    )
    assert approved_line.split()[-1] == "0"
    assert changed_line.split()[-1] == "0"
    assert f"Session: {session_path}" in result.output


def test_long_expression_is_shortened_unless_verbose(tmp_path, monkeypatch):
    expression = "(mit OR apache-2.0) AND (bsd-new OR gpl-2.0)"
    rule_path = write_rule(tmp_path, license_expression=expression)
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)
    monkeypatch.setattr(
        review_ui.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((40, 20)),
    )
    runner = CliRunner()

    concise = invoke_rule(
        runner,
        rule_path,
        "--session",
        str(tmp_path / "concise.jsonl"),
        input="n\n",
    )
    verbose = invoke_rule(
        runner,
        rule_path,
        "--session",
        str(tmp_path / "verbose.jsonl"),
        "--verbose",
        input="n\n",
    )

    assert concise.exit_code == 0, concise.output
    assert "Expression: (mit OR apache-2.0) AND (..." in concise.output
    assert f"Expression: {expression}" not in concise.output
    assert verbose.exit_code == 0, verbose.output
    assert f"Expression: {expression}" in verbose.output


def test_phrase_context_is_plain_and_respects_terminal_width(monkeypatch):
    monkeypatch.setattr(
        review_ui.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((32, 20)),
    )
    prediction = {"start_word": 5, "end_word": 6}

    context = review_ui.phrase_context(make_rule(), prediction)

    assert all(len(line) <= 32 for line in context.splitlines())
    assert "\x1b[" not in context
    assert "MIT License" in context
    assert "without restriction" not in context


def test_diff_display_handles_windows_line_endings(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    content = rule_path.read_bytes().replace(b"\r\n", b"\n")
    rule_path.write_bytes(content.replace(b"\n", b"\r\n"))
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(tmp_path / "session.jsonl"),
        "--dry-run",
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "\r" not in result.output
    assert "--- a/mit_test.RULE" in result.output
    assert "+++ b/mit_test.RULE" in result.output


def test_interactive_declines_apply_without_writing(tmp_path, monkeypatch):
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
        input="y\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert rule_path.read_bytes() == before
    assert "Apply updates to 1 rule? [y/N]" in result.output
    assert "[d] dry-run" not in result.output
    assert "Session:" in result.output
    assert "Resume:" in result.output
    assert result.output.count("\nSummary\n") == 1


def test_edit_confirmation_can_be_declined(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="e\nMIT License to use\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Change cancelled." in result.output
    _metadata, records = review.read_session(session_path)
    prediction = records[0]["predictions"][0]
    assert prediction["predicted_text"] == "MIT License"
    assert prediction["text"] == "MIT License"
    assert prediction["decision"] == review.REJECTED


def test_invalid_edit_is_not_offered_for_confirmation(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="e\nis\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "not a valid candidate" in result.output
    assert "Approve this changed phrase?" not in result.output


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
        input="y\ny\n",
    )

    assert result.exit_code == 0, result.output
    saved = Rule.from_file(str(rule_path))
    assert "{{MIT License}}" in saved.text
    assert saved.source == "ml_model"
    assert writes == [session_path]
    assert result.output.count("\nSummary\n") == 1
    assert "Run scancode-reindex-licenses" not in result.output
    written_line = next(
        line for line in result.output.splitlines() if line.startswith("Written rules:")
    )
    assert written_line.split()[-1] == "1"


def test_resumed_dry_run_reports_no_new_writes(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)
    runner = CliRunner()

    applied = invoke_rule(
        runner,
        rule_path,
        "--session",
        str(session_path),
        input="y\ny\n",
    )
    assert applied.exit_code == 0, applied.output

    resumed = runner.invoke(
        model_cli.add_model_required_phrases,
        ["--resume", str(session_path), "--dry-run"],
    )

    assert resumed.exit_code == 0, resumed.output
    assert "\nReview\n" not in resumed.output
    written_line = next(
        line for line in resumed.output.splitlines() if line.startswith("Written rules:")
    )
    assert written_line.split()[-1] == "0"
    assert "Apply updates to" not in resumed.output


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
        input="e\nMIT License to use\ny\n",
    )

    assert result.exit_code == 0, result.output
    _metadata, records = review.read_session(session_path)
    prediction = records[0]["predictions"][0]
    assert prediction["predicted_text"] == "MIT License"
    assert prediction["text"] == "MIT License to use"
    assert prediction["decision"] == review.APPROVED
    assert prediction["decision_source"] == "human"
    assert "Original prediction:" in result.output
    assert "Nearby rule context:" in result.output
    assert "Complete rule text:" not in result.output
    assert "Approve this changed phrase? [y/N]" in result.output
    assert "Proposed change:" in result.output
    counts = review_ui.session_summary(_metadata, records)
    assert counts["approved"] == 0
    assert counts["edited"] == 1


def test_interactive_unchanged_edit_is_approved_without_changed_wording(
    tmp_path,
    monkeypatch,
):
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
        input="e\nMIT License\ny\n",
    )

    assert result.exit_code == 0, result.output
    assert "Approve this phrase? [y/N]" in result.output
    assert "Approve this changed phrase?" not in result.output
    assert "Approved." in result.output
    assert "Changed and approved." not in result.output
    metadata, records = review.read_session(session_path)
    prediction = records[0]["predictions"][0]
    assert prediction["text"] == prediction["predicted_text"]
    counts = review_ui.session_summary(metadata, records)
    assert counts["approved"] == 1
    assert counts["edited"] == 0


def test_verbose_edit_shows_complete_rule_text(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(tmp_path / "session.jsonl"),
        "--verbose",
        input="e\n\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Nearby rule context:" in result.output
    assert "Complete rule text:" in result.output
    assert TEXT in result.output


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
    assert "[y] approve   [n] reject   [e] change" in result.output
    assert "[s] later     [q] save/quit   [?] help" in result.output


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
    assert "Resume:" in skipped.output
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
    assert "Dry run: no rule files were written." in resumed.output


def test_review_overview_explains_blocked_predictions(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "session.jsonl"
    use_fake_model(
        monkeypatch,
        FakePredictor(
            [
                PhrasePrediction("MIT License", 5, 6, 0.9),
                PhrasePrediction("is", 1, 1, 0.8),
            ]
        ),
    )
    use_tty(monkeypatch)

    result = invoke_rule(
        CliRunner(),
        rule_path,
        "--session",
        str(session_path),
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Selected rules:  1" in result.output
    assert "Predictions:     1 ready for review, 1 blocked by validation" in result.output
    assert "Predictions found:" not in result.output
    assert result.output.count("Prediction 1 of 1") == 1


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
    assert result.output.count("Prediction 1 of 2") == 1
    assert result.output.count("Prediction 2 of 2") == 1


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
        input="y\nx\n?\nn\nn\n",
    )

    assert result.exit_code == 0, result.output
    assert "Cannot approve with current decisions" in result.output
    second_prediction = result.output.split("Prediction 2 of 2", 1)[1]
    assert "[y] approve" not in second_prediction
    assert second_prediction.count("[n] reject   [e] change") == 2
    assert "Enter n, e, s, q, or ?." in second_prediction
    assert "Enter y, n, e, s, q, or ?." not in second_prediction
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
    assert result.output.count("\nBatch result\n") == 1
    metadata, records = review.read_session(session_path)
    first, second, third = records[0]["predictions"]
    assert (first["decision"], first["decision_source"]) == (review.APPROVED, "score")
    assert second["decision"] == review.PENDING
    assert review.prediction_needs_review(metadata, second)
    assert review.prediction_is_below_threshold(metadata, third)
    assert "Automatically approved:" in result.output
    assert "Below review threshold:" in result.output
    assert (
        "1 prediction remains pending because its score is at least --review-score "
        "and below --auto-score."
    ) in result.output
    assert (
        "1 rule was deferred because pending predictions defer the complete rule." in result.output
    )
    assert "Resume:" in result.output


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
    deferred_line = next(
        line for line in result.output.splitlines() if line.startswith("Deferred rules:")
    )
    assert deferred_line.split()[-1] == "1"
    assert "Resume:" in result.output


def test_batch_writes_valid_rule_and_blocks_protected_rule(tmp_path, monkeypatch):
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    protected_text = "License details https://spdx.org/licenses/CC-BY-NC-SA-2.0.html apply here"
    protected_path = write_rule(
        rules_directory,
        "protected.RULE",
        text=protected_text,
        ignorable_urls=["https://spdx.org/licenses/CC-BY-NC-SA-2.0.html"],
    )
    valid_path = write_rule(rules_directory, "valid.RULE")
    protected_before = protected_path.read_bytes()

    class PerRulePredictor:
        def predict(self, text):
            if text == protected_text:
                predictions = [PhrasePrediction("spdx org licenses CC BY NC SA 2 0", 3, 11, 1.0)]
            else:
                predictions = [PhrasePrediction("MIT License", 5, 6, 0.9)]
            return PredictionResult(
                words=tuple(words_from_text(text)),
                phrases=tuple(predictions),
                truncated=False,
            )

    use_fake_model(monkeypatch, PerRulePredictor())
    session_path = tmp_path / "batch.jsonl"

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
    assert protected_path.read_bytes() == protected_before
    assert "{{MIT License}}" in Rule.from_file(str(valid_path)).text
    _metadata, records = review.read_session(session_path)
    protected_record = next(
        record for record in records if record["identifier"] == "protected.RULE"
    )
    valid_record = next(record for record in records if record["identifier"] == "valid.RULE")
    protected_prediction = protected_record["predictions"][0]
    valid_prediction = valid_record["predictions"][0]
    assert protected_prediction["validation_issue"] == "rejected"
    assert protected_prediction["decision"] == review.PENDING
    assert valid_prediction["decision"] == review.APPROVED
    assert valid_prediction["decision_source"] == "score"
    assert protected_record["applied_hash"] is None
    assert valid_record["applied_hash"] is not None
    assert "Blocked by validation:" in result.output
    assert "Written rules:" in result.output


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
    assert "Ready rules were not written because --yes was not provided." in result.output
    assert "Session:" in result.output
    assert "Resume:" in result.output


def test_resumed_batch_write_does_not_report_missing_yes(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "batch.jsonl"
    use_fake_model(monkeypatch)
    use_tty(monkeypatch)
    runner = CliRunner()

    initial = invoke_rule(
        runner,
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
    assert initial.exit_code == 0, initial.output

    resumed = runner.invoke(
        model_cli.add_model_required_phrases,
        ["--resume", str(session_path)],
        input="y\n",
    )

    assert resumed.exit_code == 0, resumed.output
    assert "{{MIT License}}" in Rule.from_file(str(rule_path)).text
    assert "Written rules:           1" in resumed.output
    assert "Ready rules were not written" not in resumed.output


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
    assert "Dry run: no rule files were written." in result.output
    assert "Ready rules were not written because --yes was not provided." not in result.output
    assert "Apply updates to" not in result.output
    assert "Resume:" in result.output


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
    assert (
        "1 high-scoring prediction remains pending because truncated rule input "
        "cannot be automatically approved."
    ) in result.output


def test_batch_explains_mixed_pending_reasons(tmp_path, monkeypatch):
    rule_path = write_rule(tmp_path)
    use_fake_model(
        monkeypatch,
        FakePredictor(
            [
                PhrasePrediction("MIT License", 5, 6, 0.9),
                PhrasePrediction("without restriction", 21, 22, 0.7),
            ],
            truncated=True,
        ),
    )

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
        str(tmp_path / "batch.jsonl"),
    )

    assert result.exit_code == 0, result.output
    assert result.output.count("1 prediction remains pending") == 1
    assert result.output.count("1 high-scoring prediction remains pending") == 1
    assert (
        result.output.count(
            "1 rule was deferred because pending predictions defer the complete rule."
        )
        == 1
    )


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
    assert "remains pending because" not in result.output


@pytest.mark.parametrize("decision", [review.APPROVED, review.REJECTED])
def test_batch_explanations_ignore_decided_predictions(
    decision,
    tmp_path,
    monkeypatch,
    capsys,
):
    rule_path = write_rule(tmp_path)
    session_path = tmp_path / "batch.jsonl"
    use_fake_model(monkeypatch, FakePredictor([PhrasePrediction("MIT License", 5, 6, 0.7)]))
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
    metadata, records = review.read_session(session_path)
    records[0]["predictions"][0]["decision"] = decision
    records[0]["predictions"][0]["decision_source"] = "human"

    review_ui.print_summary(
        metadata,
        records,
        ready=0,
        deferred=0,
        unchanged=1,
        written=0,
        verbose=False,
    )

    assert "remains pending because" not in capsys.readouterr().out


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
    assert "Selected 1 rule." in result.output
    assert "Rules scanned:" not in result.output
    assert "Rules eligible:" not in result.output
    selected_line = next(
        line for line in result.output.splitlines() if line.startswith("Selected rules:")
    )
    assert selected_line.split()[-1] == "1"
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


def test_reindex_next_step_requires_an_installed_rule_write(tmp_path, monkeypatch):
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    rule_path = write_rule(rules_directory)
    rule = Rule.from_file(str(rule_path))
    monkeypatch.setattr(
        model_cli,
        "select_installed_prediction_rules",
        lambda expression: [(rule_path, rule)],
    )
    monkeypatch.setattr(model_cli, "rules_data_dir", str(rules_directory))
    monkeypatch.setattr(review, "rules_data_dir", str(rules_directory))
    use_fake_model(monkeypatch)

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        [
            "--all",
            "--model",
            "unused",
            "--batch",
            "--auto-score",
            "0.8",
            "--review-score",
            "0.5",
            "--yes",
            "--session",
            str(tmp_path / "session.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Next step:\n  scancode-reindex-licenses" in result.output


def test_reindex_next_step_is_hidden_for_installed_rule_dry_run(tmp_path, monkeypatch):
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    rule_path = write_rule(rules_directory)
    rule = Rule.from_file(str(rule_path))
    monkeypatch.setattr(
        model_cli,
        "select_installed_prediction_rules",
        lambda expression: [(rule_path, rule)],
    )
    monkeypatch.setattr(model_cli, "rules_data_dir", str(rules_directory))
    monkeypatch.setattr(review, "rules_data_dir", str(rules_directory))
    use_fake_model(monkeypatch)

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        [
            "--all",
            "--model",
            "unused",
            "--batch",
            "--auto-score",
            "0.8",
            "--review-score",
            "0.5",
            "--yes",
            "--dry-run",
            "--session",
            str(tmp_path / "session.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "scancode-reindex-licenses" not in result.output


def test_diff_styles_only_markers_on_added_lines(monkeypatch, tmp_path):
    output = []
    monkeypatch.setattr(
        review_ui.click,
        "echo",
        lambda message="", **kwargs: output.append(message),
    )
    before = b"license_expression: mit\n---\nfirst phrase and second phrase\n"
    after = b"license_expression: mit\n---\n{{first phrase}} and {{second phrase}}\n"

    review_ui.print_diff(tmp_path / "mit.RULE", before, after, color=True)

    displayed = "\n".join(output)
    expected = "\n".join(
        difflib.unified_diff(
            before.decode("utf-8").splitlines(),
            after.decode("utf-8").splitlines(),
            fromfile="a/mit.RULE",
            tofile="b/mit.RULE",
            n=3,
            lineterm="",
        )
    )
    assert click.unstyle(displayed) == expected
    assert displayed.count("\x1b[32m") == 2
    assert "\x1b[32m+" not in displayed
    assert "\x1b[31m" not in displayed
    assert "\x1b[36m" not in displayed


def test_diff_without_color_is_exact(monkeypatch, tmp_path):
    output = []
    monkeypatch.setattr(
        review_ui.click,
        "echo",
        lambda message="", **kwargs: output.append(message),
    )
    before = b"license_expression: mit\n---\nMIT License\n"
    after = b"license_expression: mit\n---\n{{MIT License}}\n"

    review_ui.print_diff(tmp_path / "mit.RULE", before, after, color=False)

    expected = list(
        difflib.unified_diff(
            before.decode("utf-8").splitlines(),
            after.decode("utf-8").splitlines(),
            fromfile="a/mit.RULE",
            tofile="b/mit.RULE",
            n=3,
            lineterm="",
        )
    )
    assert output == expected


@pytest.mark.parametrize(
    "no_color,expected",
    [(False, True), (True, False)],
)
def test_command_color_uses_tty_and_no_color(monkeypatch, no_color, expected):
    colors = []
    monkeypatch.setattr(model_cli, "stdout_is_tty", lambda: True)
    monkeypatch.setattr(model_cli, "run_new", lambda *args: colors.append(args[-1]))
    if no_color:
        monkeypatch.setenv("NO_COLOR", "1")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)

    result = CliRunner().invoke(
        model_cli.add_model_required_phrases,
        ["--all", "--model", "unused", "--predict-only"],
    )

    assert result.exit_code == 0, result.output
    assert colors == [expected]


def test_command_help_is_single_workflow():
    result = CliRunner().invoke(model_cli.add_model_required_phrases, ["--help"])

    assert result.exit_code == 0
    assert "--rule" in result.output
    assert "--rules-dir" in result.output
    assert "--all" in result.output
    assert "--predict-only" in result.output
    assert "--batch" in result.output
    assert "--resume" in result.output
    assert "--no-color" not in result.output
