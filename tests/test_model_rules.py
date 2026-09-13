# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
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
from scancode_required_phrases.model_rules import new_counts
from scancode_required_phrases.model_rules import prepare_predicted_phrases
from scancode_required_phrases.model_rules import select_rules
from scancode_required_phrases.model_rules import update_rules_from_predictions


class FakeRule:
    def __init__(self, text="some license text here"):
        self.text = text


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


def test_select_rules_reuses_scancode_selection_and_excludes_marked_rules(monkeypatch):
    calls = []

    def get_rules(**kwargs):
        calls.append(kwargs)
        return {
            "mit": [FakeRule(), FakeRule(text="under the {{mit license}} terms")],
            "bsd-new": [],
        }

    monkeypatch.setattr(model_rules, "get_updatable_rules_by_expression", get_rules)

    selected = select_rules("mit")

    assert calls == [{"license_expression": "mit", "simple_expression": False}]
    assert list(selected) == ["mit"]
    assert len(selected["mit"]) == 1


def test_select_rules_reports_an_unknown_expression(monkeypatch):
    def get_rules(**kwargs):
        raise KeyError(kwargs["license_expression"])

    monkeypatch.setattr(model_rules, "get_updatable_rules_by_expression", get_rules)
    with pytest.raises(click.ClickException, match="No rules"):
        select_rules("unknown")


def test_prepare_predicted_phrases_returns_updated_copy():
    rule = make_rule(TEXT, source="mit_1.RULE")
    counts = new_counts()

    updated_rule = prepare_predicted_phrases(
        rule=rule,
        phrases=["MIT License", "do things"],
        counts=counts,
    )

    assert updated_rule.text == (
        "Permission is granted under the {{MIT License}} to {{do things}} with this"
    )
    assert updated_rule.source == "mit_1.RULE ml_model"
    assert counts["injected"] == 2
    assert rule.text == TEXT
    assert rule.source == "mit_1.RULE"


def test_add_predicted_phrases_dry_run_does_not_mutate_rule():
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
    assert rule.text == TEXT
    assert rule.source == "mit_1.RULE"


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


def test_add_predicted_phrases_rejects_ambiguous_occurrences():
    text = "MIT License applies here. MIT License applies there."
    rule = make_rule(text)
    counts = new_counts()

    assert not add_predicted_phrases(rule, ["MIT License"], counts, dry_run=True)
    assert counts["ambiguous"] == 1
    assert rule.text == text


def test_add_predicted_phrases_rejects_an_ambiguous_shorter_phrase():
    text = "Redistribution clause and binary Redistribution clause"
    rule = make_rule(text)
    counts = new_counts()

    assert add_predicted_phrases(
        rule,
        ["Redistribution clause", "binary Redistribution clause"],
        counts,
        dry_run=True,
    )
    assert counts["ambiguous"] == 1
    assert counts["injected"] == 1
    assert rule.text == text


def test_add_predicted_phrases_rolls_back_a_conflicting_update(monkeypatch):
    rule = make_rule(TEXT)
    calls = []

    def add_phrase(rule, required_phrase, **kwargs):
        calls.append(required_phrase)
        if len(calls) == 2:
            return False
        rule.text = f"{{{{{required_phrase}}}}} " + rule.text
        return True

    monkeypatch.setattr(model_rules, "add_required_phrase_to_rule", add_phrase)
    counts = new_counts()

    assert not add_predicted_phrases(
        rule,
        ["MIT License", "do things"],
        counts,
        dry_run=True,
    )
    assert calls == ["MIT License", "do things"]
    assert counts["conflicts"] == 1
    assert counts["injected"] == 0
    assert rule.text == TEXT


def test_add_predicted_phrases_writes_exact_rule_once(tmp_path, monkeypatch):
    rule = make_rule(TEXT, source="mit_1.RULE")
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
    assert saved.text == (
        "Permission is granted under the {{MIT License}} to {{do things}} with this"
    )
    assert saved.source == "mit_1.RULE ml_model"
    assert rule.text == saved.text
    assert rule.source == saved.source


def test_add_predicted_phrases_keeps_existing_file_when_atomic_replace_fails(
    tmp_path,
    monkeypatch,
):
    rule = make_rule(TEXT)
    rule.dump(str(tmp_path))
    rule_path = tmp_path / rule.identifier
    before = rule_path.read_bytes()
    monkeypatch.setattr(model_rules, "rules_data_dir", str(tmp_path))
    monkeypatch.setattr(
        model_rules.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("simulated replace failure")),
    )

    with pytest.raises(OSError, match="simulated replace failure"):
        add_predicted_phrases(rule, ["MIT License"], new_counts())

    assert rule_path.read_bytes() == before
    assert rule.text == TEXT


def test_update_rules_from_predictions_processes_selected_rules():
    rule = make_rule(TEXT)

    counts = update_rules_from_predictions(
        selected={"mit": [rule]},
        predictor=FakePredictor(["MIT License"]),
        dry_run=True,
    )

    assert counts["rules"] == 1
    assert counts["injected"] == 1
    assert counts["changed"] == 1
    assert counts["written"] == 0
    assert rule.text == TEXT


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
    selections = []
    loads = []
    updates = []
    monkeypatch.delenv("HF_TOKEN", raising=False)

    monkeypatch.setattr(
        model_rules,
        "select_rules",
        lambda **kwargs: selections.append(kwargs) or selected,
    )
    monkeypatch.setattr(
        model_rules,
        "load_predictor",
        lambda *args, **kwargs: loads.append((args, kwargs)) or predictor,
    )

    def update(**kwargs):
        updates.append(kwargs)
        return new_counts()

    monkeypatch.setattr(model_rules, "update_rules_from_predictions", update)
    result = CliRunner().invoke(
        add_model_required_phrases,
        [
            "--model",
            str(tmp_path),
            "--license-expression",
            "mit",
            "--dry-run",
            "--limit",
            "7",
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert selections == [{"license_expression": "mit"}]
    assert loads == [
        (
            (str(tmp_path),),
            {"hf_token": None, "revision": None},
        )
    ]
    assert updates == [
        {
            "selected": selected,
            "predictor": predictor,
            "dry_run": True,
            "limit": 7,
            "verbose": True,
        }
    ]
    assert "rules written    : 0" in result.output
    assert "Dry run: no rules were saved" in result.output


def test_load_predictor_rejects_a_revision_for_a_local_model(tmp_path):
    with pytest.raises(ValueError, match="local model directory"):
        model_rules.load_predictor(tmp_path, revision="a" * 40)


def test_load_predictor_requires_a_remote_revision():
    with pytest.raises(ValueError, match="40-character"):
        model_rules.load_predictor("owner/model")


@pytest.mark.parametrize(
    "name",
    [
        "../model.safetensors",
        "models/../model.safetensors",
        r"..\model.safetensors",
        r"C:\model.safetensors",
    ],
)
def test_remote_artifact_names_reject_unsafe_paths(name):
    with pytest.raises(ValueError, match="unsafe artifact path"):
        model_rules._artifact_names({"files": {name: "digest"}})


def test_load_predictor_stages_only_declared_remote_artifacts(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    artifact = snapshot / "model.safetensors"
    artifact.write_bytes(b"weights")
    marker = {
        "files": {
            "model.safetensors": hashlib.sha256(b"weights").hexdigest(),
        }
    }
    marker_path = snapshot / "SUCCESS.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    (snapshot / ".gitattributes").write_text("metadata", encoding="utf-8")

    marker_downloads = []
    snapshot_downloads = []
    hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: marker_downloads.append(kwargs) or str(marker_path),
        snapshot_download=lambda **kwargs: snapshot_downloads.append(kwargs) or str(snapshot),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    predictor = object()

    def load(model_dir):
        assert sorted(path.name for path in model_dir.iterdir()) == [
            "SUCCESS.json",
            "model.safetensors",
        ]
        return predictor

    monkeypatch.setattr(model_rules.RequiredPhrasePredictor, "from_model_dir", load)
    revision = "a" * 40

    assert (
        model_rules.load_predictor(
            "owner/model",
            hf_token="token",
            revision=revision,
        )
        is predictor
    )
    assert marker_downloads == [
        {
            "repo_id": "owner/model",
            "filename": "SUCCESS.json",
            "revision": revision,
            "token": "token",
        }
    ]
    assert snapshot_downloads == [
        {
            "repo_id": "owner/model",
            "revision": revision,
            "token": "token",
            "allow_patterns": ["SUCCESS.json", "model.safetensors"],
        }
    ]
