# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest

from licensedcode.models import InvalidRule
from licensedcode.models import Rule

from scancode_required_phrases import model_rules
from scancode_required_phrases.inference import PhrasePrediction
from scancode_required_phrases.inference import PredictionResult


TEXT = (
    "Permission is granted under the MIT License to use copy modify merge publish "
    "distribute sublicense and sell copies of this software without restriction"
)


def make_rule(identifier="mit_test.RULE", text=TEXT, source=None, **kwargs):
    data = dict(
        identifier=identifier,
        license_expression="mit",
        text=text,
        source=source,
        is_license_notice=True,
        relevance=100,
    )
    data.update(kwargs)
    return Rule(**data)


def dump_rule(rule, directory):
    rule.dump(str(directory))
    return directory / rule.identifier


class FakePredictor:
    def __init__(self, phrases, truncated=False):
        self.phrases = phrases
        self.truncated = truncated
        self.calls = []

    def predict(self, text):
        self.calls.append(text)
        predictions = tuple(
            PhrasePrediction(
                text=phrase,
                start_word=0,
                end_word=0,
                score=0.9,
            )
            for phrase in self.phrases
        )
        return PredictionResult(
            words=tuple(text.split()),
            phrases=predictions,
            truncated=self.truncated,
        )


def test_prediction_rule_issue_matches_scancode_eligibility():
    rule = SimpleNamespace(
        is_deprecated=False,
        is_from_license=False,
        text=TEXT,
        is_approx_matchable=True,
        skip_for_required_phrase_generation=False,
    )
    assert model_rules.prediction_rule_issue(rule) is None

    rule.is_deprecated = True
    assert model_rules.prediction_rule_issue(rule) == "deprecated"
    rule.is_deprecated = False
    rule.is_from_license = True
    assert model_rules.prediction_rule_issue(rule) == "from_license"
    rule.is_from_license = False
    rule.text = "x" * 4001
    assert model_rules.prediction_rule_issue(rule) == "too_long"
    rule.text = TEXT
    rule.is_approx_matchable = False
    assert model_rules.prediction_rule_issue(rule) == "not_approx_matchable"
    rule.is_approx_matchable = True
    rule.skip_for_required_phrase_generation = True
    assert model_rules.prediction_rule_issue(rule) == "skipped"
    rule.skip_for_required_phrase_generation = False
    rule.text = "Permission under the {{MIT License}} applies"
    assert model_rules.prediction_rule_issue(rule) == "has_required_phrases"


def test_load_prediction_rule_returns_exact_resolved_path(tmp_path):
    rule_path = dump_rule(make_rule(), tmp_path)

    loaded_path, loaded_rule = model_rules.load_prediction_rule(rule_path, "mit")

    assert loaded_path == rule_path.resolve()
    assert loaded_rule.identifier == rule_path.name
    assert loaded_rule.text == TEXT


@pytest.mark.parametrize("name", ["missing.RULE", "not-a-rule.txt"])
def test_load_prediction_rule_rejects_invalid_file(tmp_path, name):
    path = tmp_path / name
    if path.suffix != ".RULE":
        path.write_text("not a rule", encoding="utf-8")

    with pytest.raises(ValueError):
        model_rules.load_prediction_rule(path)


def test_load_prediction_rule_rejects_a_directory(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        model_rules.load_prediction_rule(tmp_path)


def test_load_prediction_rule_rejects_expression_mismatch(tmp_path):
    rule_path = dump_rule(make_rule(), tmp_path)

    with pytest.raises(ValueError, match="does not use expression"):
        model_rules.load_prediction_rule(rule_path, "apache-2.0")


def test_load_prediction_rule_rejects_deprecated_rule(tmp_path):
    rule = make_rule(is_deprecated=True, relevance=0)
    rule_path = dump_rule(rule, tmp_path)

    with pytest.raises(ValueError, match="deprecated"):
        model_rules.load_prediction_rule(rule_path)


def test_load_prediction_rule_rejects_symlink(tmp_path):
    rule_path = dump_rule(make_rule(), tmp_path)
    symlink = tmp_path / "linked.RULE"
    try:
        symlink.symlink_to(rule_path)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ValueError, match="symbolic link"):
        model_rules.load_prediction_rule(symlink)


def test_load_prediction_rules_rejects_symlink_rule(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    rule_path = dump_rule(make_rule(), outside)
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    symlink = rules_directory / "linked.RULE"
    try:
        symlink.symlink_to(rule_path)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    with pytest.raises(ValueError, match="symbolic link"):
        model_rules.load_prediction_rules(rules_directory)


def test_load_prediction_rules_allows_a_symlink_root(tmp_path):
    rules_directory = tmp_path / "rules"
    rules_directory.mkdir()
    dump_rule(make_rule(), rules_directory)
    symlink = tmp_path / "rules-link"
    try:
        symlink.symlink_to(rules_directory, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    selected = model_rules.load_prediction_rules(symlink)

    assert [path for path, _rule in selected] == [
        (rules_directory / "mit_test.RULE").resolve()
    ]


def test_load_prediction_rules_is_sorted_top_level_and_filtered(tmp_path):
    dump_rule(make_rule("z.RULE"), tmp_path)
    dump_rule(make_rule("a.RULE"), tmp_path)
    dump_rule(make_rule("apache.RULE", license_expression="apache-2.0"), tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    dump_rule(make_rule("nested.RULE"), nested)

    selected = model_rules.load_prediction_rules(tmp_path, "mit")

    assert [path.name for path, _rule in selected] == ["a.RULE", "z.RULE"]


def test_load_prediction_rules_validates_the_complete_set_once(tmp_path, monkeypatch):
    dump_rule(make_rule("first.RULE"), tmp_path)
    dump_rule(make_rule("second.RULE"), tmp_path)
    calls = []
    licenses = object()
    monkeypatch.setattr(model_rules, "get_licenses_db", lambda: licenses)
    monkeypatch.setattr(
        model_rules,
        "validate_rules",
        lambda **kwargs: calls.append(kwargs),
    )

    model_rules.load_prediction_rules(tmp_path)

    assert len(calls) == 1
    assert calls[0]["licenses_by_key"] is licenses
    assert [rule.identifier for rule in calls[0]["rules"]] == [
        "first.RULE",
        "second.RULE",
    ]


def test_load_prediction_rules_rejects_duplicate_identifiers(tmp_path, monkeypatch):
    first_path = tmp_path / "first.RULE"
    second_path = tmp_path / "second.RULE"
    first_path.touch()
    second_path.touch()
    rule = make_rule("duplicate.RULE")

    monkeypatch.setattr(
        model_rules,
        "_load_rule_file",
        lambda path: (Path(path).resolve(), rule),
    )

    with pytest.raises(ValueError, match="duplicate identifiers"):
        model_rules.load_prediction_rules(tmp_path)


def test_load_prediction_rules_aborts_for_a_malformed_rule(tmp_path):
    dump_rule(make_rule("good.RULE"), tmp_path)
    (tmp_path / "bad.RULE").write_text("not frontmatter", encoding="utf-8")

    with pytest.raises(InvalidRule):
        model_rules.load_prediction_rules(tmp_path)


def test_load_prediction_rules_omits_ineligible_rules(tmp_path):
    dump_rule(make_rule("eligible.RULE"), tmp_path)
    dump_rule(make_rule("marked.RULE", text="The {{MIT License}} applies here"), tmp_path)

    selected = model_rules.load_prediction_rules(tmp_path)

    assert [path.name for path, _rule in selected] == ["eligible.RULE"]


def test_select_installed_prediction_rules_reuses_scancode_selection(
    tmp_path,
    monkeypatch,
):
    first = make_rule("first.RULE")
    second = make_rule("second.RULE", text="The {{MIT License}} applies here")
    dump_rule(first, tmp_path)
    dump_rule(second, tmp_path)
    calls = []

    def get_rules(**kwargs):
        calls.append(kwargs)
        return {"mit": [second, first]}

    monkeypatch.setattr(model_rules, "get_updatable_rules_by_expression", get_rules)
    monkeypatch.setattr(model_rules, "rules_data_dir", str(tmp_path))

    selected = model_rules.select_installed_prediction_rules("mit")

    assert calls == [{"license_expression": "mit", "simple_expression": False}]
    assert [(path.name, rule.identifier) for path, rule in selected] == [
        ("first.RULE", "first.RULE")
    ]


def test_select_installed_prediction_rules_reports_unknown_expression(monkeypatch):
    def get_rules(**kwargs):
        raise KeyError(kwargs["license_expression"])

    monkeypatch.setattr(model_rules, "get_updatable_rules_by_expression", get_rules)

    with pytest.raises(ValueError, match="No rules"):
        model_rules.select_installed_prediction_rules("unknown")


def test_predict_rule_candidates_returns_validation_issues():
    rule = make_rule(text="MIT License applies here. MIT License applies there.")
    predictor = FakePredictor(["MIT License", "is"])

    result, candidates = model_rules.predict_rule_candidates(rule, predictor)

    assert result.truncated is False
    assert predictor.calls == [rule.text]
    assert [(prediction.text, issue) for prediction, issue in candidates] == [
        ("MIT License", "ambiguous"),
        ("is", "rejected"),
    ]


def test_predict_rule_candidates_preserves_truncation():
    predictor = FakePredictor([], truncated=True)

    result, candidates = model_rules.predict_rule_candidates(make_rule(), predictor)

    assert result.truncated is True
    assert candidates == ()


def test_prepare_predicted_phrases_returns_updated_copy():
    rule = make_rule(source="mit_1.RULE")

    updated = model_rules.prepare_predicted_phrases(
        rule,
        ["MIT License", "without restriction"],
    )

    assert "{{MIT License}}" in updated.text
    assert "{{without restriction}}" in updated.text
    assert updated.source == "mit_1.RULE ml_model"
    assert rule.text == TEXT
    assert rule.source == "mit_1.RULE"


def test_prepare_predicted_phrases_does_not_duplicate_source():
    rule = make_rule(source="mit_1.RULE ml_model")

    updated = model_rules.prepare_predicted_phrases(rule, ["MIT License"])

    assert updated.source == "mit_1.RULE ml_model"


def test_prepare_predicted_phrases_rejects_duplicate_text():
    with pytest.raises(ValueError, match="Duplicate"):
        model_rules.prepare_predicted_phrases(
            make_rule(),
            ["MIT License", "MIT License"],
        )


def test_prepare_predicted_phrases_rejects_an_invalid_phrase():
    with pytest.raises(ValueError, match="not safe"):
        model_rules.prepare_predicted_phrases(make_rule(), ["is"])


def test_prepare_predicted_phrases_adds_longest_phrase_first(monkeypatch):
    calls = []

    def add_phrase(rule, required_phrase, **kwargs):
        calls.append(required_phrase)
        rule.text = f"{rule.text} {required_phrase}"
        return True

    monkeypatch.setattr(model_rules, "candidate_issue", lambda rule, phrase: None)
    monkeypatch.setattr(model_rules, "add_required_phrase_to_rule", add_phrase)

    model_rules.prepare_predicted_phrases(
        make_rule(),
        ["MIT License", "Permission under the MIT License"],
    )

    assert calls == ["Permission under the MIT License", "MIT License"]


def test_prepare_predicted_phrases_rejects_complete_conflict():
    rule = make_rule(text="binary Redistribution clause applies to these software copies")

    with pytest.raises(ValueError, match="conflict"):
        model_rules.prepare_predicted_phrases(
            rule,
            ["Redistribution clause", "binary Redistribution clause"],
        )
    assert rule.text == "binary Redistribution clause applies to these software copies"


def test_serialize_rule_returns_scancode_dump_bytes(tmp_path):
    rule = make_rule()
    rule_path = dump_rule(rule, tmp_path)
    updated = model_rules.prepare_predicted_phrases(rule, ["MIT License"])

    content = model_rules.serialize_rule(updated, rule_path)
    expected_dir = tmp_path / "expected"
    expected_dir.mkdir()
    updated.dump(str(expected_dir))

    assert content == (expected_dir / rule.identifier).read_bytes()


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_serialize_rule_preserves_line_endings(tmp_path, newline):
    rule_path = dump_rule(make_rule(), tmp_path)
    original = rule_path.read_bytes().replace(b"\r\n", b"\n")
    rule_path.write_bytes(original.replace(b"\n", newline))
    rule = Rule.from_file(str(rule_path))
    updated = model_rules.prepare_predicted_phrases(rule, ["MIT License"])

    content = model_rules.serialize_rule(updated, rule_path)

    if newline == b"\r\n":
        assert b"\r\n" in content
        assert b"\n" not in content.replace(b"\r\n", b"")
    else:
        assert b"\r\n" not in content


def test_serialize_rule_rejects_identifier_path_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        model_rules.serialize_rule(make_rule(), tmp_path / "other.RULE")


def test_write_rule_atomically_writes_exact_path_and_preserves_mode(tmp_path):
    rule = make_rule()
    rule_path = dump_rule(rule, tmp_path)
    original_mode = stat.S_IMODE(rule_path.stat().st_mode)
    updated = model_rules.prepare_predicted_phrases(rule, ["MIT License"])
    content = model_rules.serialize_rule(updated, rule_path)

    written_hash = model_rules.write_rule_atomically(
        rule_path,
        content,
        model_rules.file_sha256(rule_path),
    )

    assert rule_path.read_bytes() == content
    assert written_hash == hashlib.sha256(content).hexdigest()
    assert stat.S_IMODE(rule_path.stat().st_mode) == original_mode
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_rule_atomically_rejects_a_stale_rule(tmp_path):
    rule_path = dump_rule(make_rule(), tmp_path)
    before = rule_path.read_bytes()

    with pytest.raises(ValueError, match="changed"):
        model_rules.write_rule_atomically(rule_path, b"replacement", "0" * 64)

    assert rule_path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_rule_atomically_detects_change_before_replace(tmp_path, monkeypatch):
    rule_path = dump_rule(make_rule(), tmp_path)
    expected_hash = model_rules.file_sha256(rule_path)
    real_file_sha256 = model_rules.file_sha256

    def change_rule(path):
        Path(path).write_bytes(b"changed outside this process")
        return real_file_sha256(path)

    monkeypatch.setattr(model_rules, "file_sha256", change_rule)

    with pytest.raises(ValueError, match="changed"):
        model_rules.write_rule_atomically(rule_path, b"replacement", expected_hash)

    assert rule_path.read_bytes() == b"changed outside this process"
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_rule_atomically_keeps_file_when_replace_fails(tmp_path, monkeypatch):
    rule_path = dump_rule(make_rule(), tmp_path)
    before = rule_path.read_bytes()
    monkeypatch.setattr(
        model_rules.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("simulated replace failure")),
    )

    with pytest.raises(OSError, match="simulated replace failure"):
        model_rules.write_rule_atomically(
            rule_path,
            b"replacement",
            model_rules.file_sha256(rule_path),
        )

    assert rule_path.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


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


def test_load_predictor_resolves_snapshot_symlinks(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    blobs = tmp_path / "blobs"
    snapshot.mkdir()
    blobs.mkdir()
    marker = {"files": {"model.safetensors": hashlib.sha256(b"weights").hexdigest()}}
    marker_blob = blobs / "marker"
    marker_blob.write_text(json.dumps(marker), encoding="utf-8")
    weights_blob = blobs / "weights"
    weights_blob.write_bytes(b"weights")
    try:
        (snapshot / "SUCCESS.json").symlink_to(marker_blob)
        (snapshot / "model.safetensors").symlink_to(weights_blob)
    except OSError:
        pytest.skip("symbolic links are unavailable")

    hub = SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(snapshot / "SUCCESS.json"),
        snapshot_download=lambda **kwargs: str(snapshot),
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    def load(model_dir):
        for name in ("SUCCESS.json", "model.safetensors"):
            path = model_dir / name
            assert path.is_file()
            assert not path.is_symlink()
        return "predictor"

    monkeypatch.setattr(model_rules.RequiredPhrasePredictor, "from_model_dir", load)

    assert model_rules.load_predictor("owner/model", revision="a" * 40) == "predictor"


def test_load_predictor_stages_only_declared_remote_artifacts(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    artifact = snapshot / "model.safetensors"
    artifact.write_bytes(b"weights")
    marker = {"files": {"model.safetensors": hashlib.sha256(b"weights").hexdigest()}}
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

    assert model_rules.load_predictor(
        "owner/model",
        hf_token="token",
        revision=revision,
    ) is predictor
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
