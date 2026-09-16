# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from licensedcode.models import Rule

from scancode_required_phrases import review
from scancode_required_phrases.inference import PhrasePrediction
from scancode_required_phrases.model_rules import file_sha256


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


def make_prediction(text="MIT License", start=5, end=6, score=0.9):
    return PhrasePrediction(
        text=text,
        start_word=start,
        end_word=end,
        score=score,
    )


def make_session(tmp_path, run_mode="interactive", score=0.9):
    rule = make_rule()
    rule.dump(str(tmp_path))
    rule_path = (tmp_path / rule.identifier).resolve()
    metadata = review.create_metadata(
        model="model-dir",
        model_revision=None,
        target_mode="rule",
        target=rule_path,
        run_mode=run_mode,
        rules_scanned=1,
        rules_eligible=1,
        truncated_rules=0,
        auto_score=0.8 if run_mode == "batch" else None,
        review_score=0.5 if run_mode == "batch" else None,
    )
    prediction = review.create_prediction(make_prediction(score=score))
    record = review.create_rule_record(
        rule_path=rule_path,
        rule=rule,
        predictions=[prediction],
        truncated=False,
    )
    return metadata, [record], rule_path


def approve(prediction, source="human"):
    prediction["decision"] = review.APPROVED
    prediction["decision_source"] = source


def test_session_round_trip(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    session_path = tmp_path / "session.jsonl"

    review.write_session(session_path, metadata, records)

    assert review.read_session(session_path) == (metadata, records)
    assert not list(tmp_path.glob(".session.jsonl.*.tmp"))


def test_large_session_round_trip(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    metadata["target_mode"] = "rules_dir"
    metadata["target"] = str(tmp_path.resolve())
    metadata["rules_scanned"] = 500
    metadata["rules_eligible"] = 500
    template = records[0]
    records = []
    for index in range(500):
        identifier = f"mit_{index}.RULE"
        record = dict(template)
        record["path"] = str((tmp_path / identifier).resolve())
        record["identifier"] = identifier
        record["predictions"] = [dict(template["predictions"][0])]
        records.append(record)
    session_path = tmp_path / "large.jsonl"

    review.write_session(session_path, metadata, records)
    loaded_metadata, loaded_records = review.read_session(session_path)

    assert loaded_metadata == metadata
    assert len(loaded_records) == 500
    assert loaded_records[-1]["identifier"] == "mit_499.RULE"


def test_write_session_keeps_existing_file_when_replace_fails(tmp_path, monkeypatch):
    metadata, records, _rule_path = make_session(tmp_path)
    session_path = tmp_path / "session.jsonl"
    review.write_session(session_path, metadata, records)
    before = session_path.read_bytes()
    monkeypatch.setattr(
        review.os,
        "replace",
        lambda *args: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        review.write_session(session_path, metadata, records)

    assert session_path.read_bytes() == before
    assert not list(tmp_path.glob(".session.jsonl.*.tmp"))


def test_read_session_rejects_old_schema_and_malformed_input(tmp_path):
    old_session = tmp_path / "old.jsonl"
    old_session.write_text(json.dumps({"format_version": 0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fields"):
        review.read_session(old_session)

    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text("{bad}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1: malformed JSON"):
        review.read_session(malformed)

    invalid_utf8 = tmp_path / "invalid.jsonl"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="Cannot read session"):
        review.read_session(invalid_utf8)


def test_validate_session_rejects_unknown_fields(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    records[0]["extra"] = True

    with pytest.raises(ValueError, match="invalid fields"):
        review.validate_session(metadata, records, "session.jsonl")


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda metadata, prediction, record: prediction.update(score=float("nan")), "score"),
        (
            lambda metadata, prediction, record: prediction.update(
                validation_issue="ambiguous",
                decision="approved",
                decision_source="human",
            ),
            "invalid predictions",
        ),
        (
            lambda metadata, prediction, record: prediction.update(
                decision="pending",
                decision_source="human",
            ),
            "pending prediction",
        ),
        (
            lambda metadata, prediction, record: prediction.update(
                decision="approved",
                decision_source=None,
            ),
            "no decision source",
        ),
        (
            lambda metadata, prediction, record: prediction.update(
                decision="rejected",
                decision_source="score",
            ),
            "human decision",
        ),
        (
            lambda metadata, prediction, record: record.update(
                expected_hash=record["original_hash"]
            ),
            "equals original",
        ),
        (
            lambda metadata, prediction, record: record.update(applied_hash="a" * 64),
            "does not match expected",
        ),
    ],
)
def test_validate_session_rejects_inconsistent_state(tmp_path, change, error):
    metadata, records, _rule_path = make_session(tmp_path)
    change(metadata, records[0]["predictions"][0], records[0])

    with pytest.raises(ValueError, match=error):
        review.validate_session(metadata, records, "session.jsonl")


def test_validate_session_rejects_invalid_score_approval(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path, run_mode="batch", score=0.7)
    approve(records[0]["predictions"][0], source="score")

    with pytest.raises(ValueError, match="below the automatic threshold"):
        review.validate_session(metadata, records, "session.jsonl")


def test_validate_session_rejects_truncated_score_approval(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path, run_mode="batch")
    records[0]["truncated"] = True
    approve(records[0]["predictions"][0], source="score")

    with pytest.raises(ValueError, match="truncated"):
        review.validate_session(metadata, records, "session.jsonl")


def test_validate_session_rejects_edited_score_approval(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path, run_mode="batch")
    prediction = records[0]["predictions"][0]
    prediction["text"] = "MIT License to use"
    approve(prediction, source="score")

    with pytest.raises(ValueError, match="cannot be edited"):
        review.validate_session(metadata, records, "session.jsonl")


def test_validate_session_rejects_below_threshold_decision(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path, run_mode="batch", score=0.4)
    approve(records[0]["predictions"][0])

    with pytest.raises(ValueError, match="below-threshold"):
        review.validate_session(metadata, records, "session.jsonl")


def test_validate_session_rejects_duplicate_paths_and_identifiers(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    duplicate = dict(records[0])
    duplicate["predictions"] = [dict(records[0]["predictions"][0])]

    with pytest.raises(ValueError, match="duplicate rule path"):
        review.validate_session(metadata, [records[0], duplicate], "session.jsonl")

    metadata["target_mode"] = "rules_dir"
    metadata["target"] = str(tmp_path.resolve())
    records[0]["path"] = str((tmp_path / "first" / "mit_test.RULE").resolve())
    duplicate["path"] = str((tmp_path / "second" / "mit_test.RULE").resolve())
    with pytest.raises(ValueError, match="duplicate rule identifier"):
        review.validate_session(metadata, [records[0], duplicate], "session.jsonl")


def test_validate_session_rejects_duplicate_predictions(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    prediction = dict(records[0]["predictions"][0])
    records[0]["predictions"].append(prediction)

    with pytest.raises(ValueError, match="duplicate prediction"):
        review.validate_session(metadata, records, "session.jsonl")


def test_create_session_path_reserves_default_path(tmp_path, monkeypatch):
    monkeypatch.setattr(review.click, "get_app_dir", lambda *args, **kwargs: str(tmp_path))

    session_path = review.create_session_path()

    assert session_path.parent == (tmp_path / "sessions").resolve()
    assert session_path.name.startswith("model-required-phrases-")
    assert session_path.suffix == ".jsonl"
    assert session_path.is_file()
    assert session_path.stat().st_size == 0


def test_create_session_path_refuses_explicit_collision(tmp_path):
    session_path = tmp_path / "session.jsonl"
    session_path.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        review.create_session_path(session_path)

    assert session_path.read_text(encoding="utf-8") == "keep"


def test_validate_session_rejects_expected_hash_without_approval(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    records[0]["predictions"][0]["decision"] = review.REJECTED
    records[0]["predictions"][0]["decision_source"] = "human"
    records[0]["expected_hash"] = "a" * 64

    with pytest.raises(ValueError, match="no approved predictions"):
        review.validate_session(metadata, records, "session.jsonl")


def test_create_session_path_creates_explicit_parent(tmp_path):
    session_path = tmp_path / "new" / "session.jsonl"

    created = review.create_session_path(session_path)

    assert created == session_path.resolve()
    assert created.is_file()
    assert created.stat().st_size == 0


def test_validate_session_paths_rejects_path_outside_directory(tmp_path):
    root = (tmp_path / "rules").resolve()
    root.mkdir()
    metadata, records, _rule_path = make_session(tmp_path)
    metadata["target_mode"] = "rules_dir"
    metadata["target"] = str(root)

    with pytest.raises(ValueError, match="outside the session target"):
        review.validate_session_paths(metadata, records)


def test_validate_session_paths_rejects_symlink_before_hashing(tmp_path, monkeypatch):
    metadata, records, rule_path = make_session(tmp_path)
    target = tmp_path / "target.RULE"
    rule_path.rename(target)
    try:
        rule_path.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    monkeypatch.setattr(
        review,
        "file_sha256",
        lambda path: pytest.fail("path must be rejected before hashing"),
    )

    with pytest.raises(ValueError, match="symbolic link"):
        review.load_session_rules(metadata, records)


def test_load_session_rules_loads_unchanged_rule(tmp_path):
    metadata, records, rule_path = make_session(tmp_path)

    loaded, reconciled = review.load_session_rules(metadata, records)

    assert list(loaded) == [str(rule_path)]
    assert loaded[str(rule_path)].identifier == rule_path.name
    assert reconciled == []


def expected_rule_content(record, rule_path):
    rule = Rule.from_file(str(rule_path))
    updated = review.prepare_predicted_phrases(rule, ["MIT License"])
    return review.serialize_rule(updated, rule_path)


def test_load_session_rules_reconciles_interrupted_session_update(tmp_path):
    metadata, records, rule_path = make_session(tmp_path)
    approve(records[0]["predictions"][0])
    content = expected_rule_content(records[0], rule_path)
    expected_hash = hashlib.sha256(content).hexdigest()
    records[0]["expected_hash"] = expected_hash
    rule_path.write_bytes(content)

    loaded, reconciled = review.load_session_rules(metadata, records)

    assert loaded == {}
    assert reconciled == ["mit_test.RULE"]
    assert records[0]["applied_hash"] == expected_hash


def test_load_session_rules_accepts_applied_rule(tmp_path):
    metadata, records, rule_path = make_session(tmp_path)
    approve(records[0]["predictions"][0])
    content = expected_rule_content(records[0], rule_path)
    expected_hash = hashlib.sha256(content).hexdigest()
    records[0]["expected_hash"] = expected_hash
    records[0]["applied_hash"] = expected_hash
    rule_path.write_bytes(content)

    loaded, reconciled = review.load_session_rules(metadata, records)

    assert loaded == {}
    assert reconciled == []


def test_load_session_rules_rejects_reverted_applied_rule(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    approve(records[0]["predictions"][0])
    records[0]["expected_hash"] = "a" * 64
    records[0]["applied_hash"] = "a" * 64

    with pytest.raises(ValueError, match="reverted"):
        review.load_session_rules(metadata, records)


def test_load_session_rules_rejects_other_stale_content(tmp_path):
    metadata, records, rule_path = make_session(tmp_path)
    rule_path.write_bytes(b"changed")

    with pytest.raises(ValueError, match="stale"):
        review.load_session_rules(metadata, records)


def test_load_session_rules_rejects_changed_prediction_offsets(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    records[0]["predictions"][0]["start_word"] = 0
    records[0]["predictions"][0]["end_word"] = 1

    with pytest.raises(ValueError, match="offsets"):
        review.load_session_rules(metadata, records)


def test_prepare_rule_updates_defers_pending_rule(tmp_path):
    metadata, records, _rule_path = make_session(tmp_path)
    loaded, _reconciled = review.load_session_rules(metadata, records)

    work, unchanged, deferred = review.prepare_rule_updates(metadata, records, loaded)

    assert work == []
    assert unchanged == []
    assert deferred == ["mit_test.RULE"]


def test_prepare_rule_updates_keeps_pending_rule_untouched(tmp_path):
    first = make_rule("first.RULE")
    second = make_rule("second.RULE")
    first.dump(str(tmp_path))
    second.dump(str(tmp_path))
    first_path = (tmp_path / first.identifier).resolve()
    second_path = (tmp_path / second.identifier).resolve()
    metadata = review.create_metadata(
        model="model-dir",
        model_revision=None,
        target_mode="rules_dir",
        target=tmp_path.resolve(),
        run_mode="batch",
        rules_scanned=2,
        rules_eligible=2,
        truncated_rules=0,
        auto_score=0.8,
        review_score=0.5,
    )
    first_prediction = review.create_prediction(make_prediction())
    approve(first_prediction, source="score")
    records = [
        review.create_rule_record(first_path, first, [first_prediction], False),
        review.create_rule_record(
            second_path,
            second,
            [review.create_prediction(make_prediction(score=0.7))],
            False,
        ),
    ]
    loaded, _reconciled = review.load_session_rules(metadata, records)

    work, unchanged, deferred = review.prepare_rule_updates(metadata, records, loaded)

    assert unchanged == []
    assert deferred == ["second.RULE"]
    assert [record["identifier"] for record, _path, _content in work] == ["first.RULE"]
    session_path = tmp_path / "session.jsonl"
    review.write_rule_updates(session_path, metadata, records, work)
    assert "{{MIT License}}" in Rule.from_file(str(first_path)).text
    assert "{{" not in Rule.from_file(str(second_path)).text


def test_prepare_rule_updates_uses_only_approved_phrases(tmp_path):
    metadata, records, rule_path = make_session(tmp_path)
    rejected = review.create_prediction(make_prediction("without restriction", 21, 22, 0.7))
    rejected["decision"] = review.REJECTED
    rejected["decision_source"] = "human"
    records[0]["predictions"].append(rejected)
    approve(records[0]["predictions"][0])
    loaded, _reconciled = review.load_session_rules(metadata, records)

    work, unchanged, deferred = review.prepare_rule_updates(metadata, records, loaded)

    assert unchanged == []
    assert deferred == []
    assert len(work) == 1
    _record, prepared_path, content = work[0]
    assert prepared_path == rule_path
    saved = tmp_path / "saved"
    saved.mkdir()
    (saved / rule_path.name).write_bytes(content)
    updated = Rule.from_file(str(saved / rule_path.name))
    assert "{{MIT License}}" in updated.text
    assert "{{without restriction}}" not in updated.text
    assert records[0]["expected_hash"] == hashlib.sha256(content).hexdigest()


def test_prepare_rule_updates_clears_old_expected_hash_for_noop(
    tmp_path,
    monkeypatch,
):
    metadata, records, rule_path = make_session(tmp_path)
    approve(records[0]["predictions"][0])
    records[0]["expected_hash"] = "a" * 64
    loaded, _reconciled = review.load_session_rules(metadata, records)
    monkeypatch.setattr(
        review,
        "serialize_rule",
        lambda rule, path: rule_path.read_bytes(),
    )

    work, unchanged, deferred = review.prepare_rule_updates(metadata, records, loaded)

    assert work == []
    assert deferred == []
    assert unchanged == ["mit_test.RULE"]
    assert records[0]["expected_hash"] is None
    assert records[0]["applied_hash"] is None


def test_write_rule_updates_saves_expected_hash_before_writing(tmp_path, monkeypatch):
    metadata, records, _rule_path = make_session(tmp_path)
    approve(records[0]["predictions"][0])
    loaded, _reconciled = review.load_session_rules(metadata, records)
    work, _unchanged, _deferred = review.prepare_rule_updates(metadata, records, loaded)
    session_path = tmp_path / "session.jsonl"
    writes = []

    monkeypatch.setattr(
        review,
        "write_rule_atomically",
        lambda **kwargs: writes.append(kwargs) or records[0]["expected_hash"],
    )

    review.write_rule_updates(session_path, metadata, records, work)

    assert len(writes) == 1
    saved_metadata, saved_records = review.read_session(session_path)
    assert saved_metadata == metadata
    assert saved_records[0]["applied_hash"] == saved_records[0]["expected_hash"]


def test_write_rule_updates_does_not_write_if_initial_session_save_fails(
    tmp_path,
    monkeypatch,
):
    metadata, records, _rule_path = make_session(tmp_path)
    approve(records[0]["predictions"][0])
    loaded, _reconciled = review.load_session_rules(metadata, records)
    work, _unchanged, _deferred = review.prepare_rule_updates(metadata, records, loaded)
    monkeypatch.setattr(
        review,
        "write_session",
        lambda *args: (_ for _ in ()).throw(OSError("session save failed")),
    )
    monkeypatch.setattr(
        review,
        "write_rule_atomically",
        lambda **kwargs: pytest.fail("rule must not be written"),
    )

    with pytest.raises(OSError, match="session save failed"):
        review.write_rule_updates(tmp_path / "session.jsonl", metadata, records, work)


def test_interrupted_session_save_after_rule_write_is_reconciled(tmp_path, monkeypatch):
    metadata, records, rule_path = make_session(tmp_path)
    approve(records[0]["predictions"][0])
    loaded, _reconciled = review.load_session_rules(metadata, records)
    work, _unchanged, _deferred = review.prepare_rule_updates(metadata, records, loaded)
    session_path = tmp_path / "session.jsonl"
    real_write_session = review.write_session
    calls = 0

    def fail_second_save(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("session save failed")
        real_write_session(*args)

    monkeypatch.setattr(review, "write_session", fail_second_save)

    with pytest.raises(OSError, match="session save failed"):
        review.write_rule_updates(session_path, metadata, records, work)

    saved_metadata, saved_records = review.read_session(session_path)
    assert saved_records[0]["applied_hash"] is None
    assert file_sha256(rule_path) == saved_records[0]["expected_hash"]

    loaded, reconciled = review.load_session_rules(saved_metadata, saved_records)
    assert loaded == {}
    assert reconciled == ["mit_test.RULE"]
