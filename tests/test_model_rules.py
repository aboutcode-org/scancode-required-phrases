# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import click
from click.testing import CliRunner
import pytest

from licensedcode.models import Rule

from scancode_required_phrases import model_rules
from scancode_required_phrases.inference import PhrasePrediction
from scancode_required_phrases.inference import PredictionResult
from scancode_required_phrases.model_rules import add_model_required_phrases
from scancode_required_phrases.model_rules import add_predicted_phrases
from scancode_required_phrases.model_rules import is_updatable
from scancode_required_phrases.model_rules import new_counts
from scancode_required_phrases.model_rules import select_rules
from scancode_required_phrases.model_rules import update_rules_from_predictions


class FakeRule:
    def __init__(
        self,
        text="some license text here",
        is_from_license=False,
        is_approx_matchable=True,
        skip=False,
    ):
        self.text = text
        self.is_from_license = is_from_license
        self.is_approx_matchable = is_approx_matchable
        self.skip_for_required_phrase_generation = skip


class FakePredictor:
    def __init__(self, phrases, truncated=False):
        self.phrases = phrases
        self.truncated = truncated

    def predict(self, text):
        predictions = tuple(
            PhrasePrediction(
                text=phrase,
                start_word=0,
                end_word=0,
                confidence=1.0,
            )
            for phrase in self.phrases
        )
        return PredictionResult(
            words=tuple(text.split()),
            phrases=predictions,
            truncated=self.truncated,
        )


def make_rule(text, source=None):
    rule = Rule(
        license_expression="mit",
        identifier="mit_test.RULE",
        text=text,
        source=source,
        is_license_reference=True,
        relevance=100,
    )
    return rule


TEXT = "Permission is granted under the MIT License to do things with this"


@pytest.mark.parametrize(
    "rule",
    [
        FakeRule(is_from_license=True),
        FakeRule(text="x" * 4001),
        FakeRule(is_approx_matchable=False),
        FakeRule(skip=True),
        FakeRule(text="under the {{mit license}} terms"),
    ],
)
def test_is_updatable_excludes_ineligible_rules(rule):
    assert not is_updatable(rule)


def test_is_updatable_accepts_a_plain_rule():
    assert is_updatable(FakeRule())


def test_select_rules_filters_and_groups_rules(monkeypatch):
    monkeypatch.setattr(
        model_rules,
        "get_base_rules_by_expression",
        lambda expression: {
            "mit": [FakeRule(), FakeRule(is_from_license=True)],
            "bsd-new": [FakeRule(skip=True)],
        },
    )

    selected = select_rules()

    assert list(selected) == ["mit"]
    assert len(selected["mit"]) == 1


def test_select_rules_reports_an_unknown_expression(monkeypatch):
    def get_rules(expression):
        raise KeyError(expression)

    monkeypatch.setattr(model_rules, "get_base_rules_by_expression", get_rules)
    with pytest.raises(click.ClickException, match="No rules"):
        select_rules("unknown")


def test_add_predicted_phrases_marks_all_candidates_and_preserves_source():
    rule = make_rule(TEXT, source="mit_1.RULE")
    counts = new_counts()

    updated = add_predicted_phrases(
        rule=rule,
        phrases=["MIT License", "do things"],
        counts=counts,
        dry_run=True,
    )

    assert updated
    assert counts["injected"] == 2
    assert "{{MIT License}}" in rule.text
    assert "{{do things}}" in rule.text
    assert rule.source == "mit_1.RULE ml_model"


def test_add_predicted_phrases_rejects_an_unsuitable_candidate():
    rule = make_rule(TEXT)
    counts = new_counts()

    assert not add_predicted_phrases(rule, ["is"], counts, dry_run=True)
    assert counts["rejected"] == 1
    assert "{{" not in rule.text


def test_add_predicted_phrases_counts_a_candidate_not_in_the_rule():
    rule = make_rule(TEXT)
    counts = new_counts()

    assert not add_predicted_phrases(
        rule,
        ["Apache License"],
        counts,
        dry_run=True,
    )
    assert counts["not_found"] == 1


def test_add_predicted_phrases_writes_once(tmp_path, monkeypatch):
    rule = make_rule(TEXT)
    original_dump = Rule.dump
    writes = []

    def dump(rule, rules_data_dir):
        writes.append(rule.identifier)
        original_dump(rule, rules_data_dir)

    monkeypatch.setattr(Rule, "dump", dump)
    monkeypatch.setattr(model_rules, "rules_data_dir", str(tmp_path))

    assert add_predicted_phrases(rule, ["MIT License", "do things"], new_counts())
    saved = Rule.from_file(str(tmp_path / rule.identifier))
    assert writes == [rule.identifier]
    assert "{{MIT License}}" in saved.text
    assert "{{do things}}" in saved.text


def test_update_rules_from_predictions_processes_selected_rules():
    rule = make_rule(TEXT)

    counts = update_rules_from_predictions(
        selected={"mit": [rule]},
        predictor=FakePredictor(["MIT License"]),
        dry_run=True,
    )

    assert counts["rules"] == 1
    assert counts["injected"] == 1
    assert counts["written"] == 1


def test_update_rules_from_predictions_counts_truncation_and_honors_limit():
    rules = [make_rule(TEXT) for _ in range(3)]

    counts = update_rules_from_predictions(
        selected={"mit": rules},
        predictor=FakePredictor([], truncated=True),
        dry_run=True,
        limit=2,
    )

    assert counts["rules"] == 2
    assert counts["truncated"] == 2


def test_command_does_not_load_a_model_without_eligible_rules(monkeypatch):
    monkeypatch.setattr(model_rules, "select_rules", lambda **kwargs: {})

    def fail(*args, **kwargs):
        raise AssertionError("model should not load")

    monkeypatch.setattr(model_rules, "load_predictor", fail)
    result = CliRunner().invoke(add_model_required_phrases, ["--model", "unused"])

    assert result.exit_code == 0
    assert "No eligible rules found" in result.output


def test_command_wires_selection_prediction_and_dry_run(monkeypatch, tmp_path):
    selected = {"mit": [object()]}
    predictor = object()
    calls = []
    monkeypatch.setattr(model_rules, "select_rules", lambda **kwargs: selected)
    monkeypatch.setattr(model_rules, "load_predictor", lambda *args, **kwargs: predictor)

    def update(**kwargs):
        calls.append(kwargs)
        return new_counts()

    monkeypatch.setattr(model_rules, "update_rules_from_predictions", update)
    result = CliRunner().invoke(
        add_model_required_phrases,
        ["--model", str(tmp_path), "--license-expression", "mit", "--dry-run"],
    )

    assert result.exit_code == 0
    assert calls == [
        {
            "selected": selected,
            "predictor": predictor,
            "dry_run": True,
            "limit": 0,
            "verbose": False,
        }
    ]


def test_load_predictor_downloads_a_hugging_face_snapshot(tmp_path, monkeypatch):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    downloads = []
    predictor = object()
    hub = SimpleNamespace(
        snapshot_download=lambda repo_id, token: downloads.append((repo_id, token))
        or str(model_dir)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(
        model_rules.RequiredPhrasePredictor,
        "from_model_dir",
        lambda path: predictor,
    )

    assert model_rules.load_predictor("owner/model", "token") is predictor
    assert downloads == [("owner/model", "token")]
