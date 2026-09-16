# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Store and apply reviewed model predictions."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import click

from licensedcode.models import Rule
from licensedcode.models import rules_data_dir

from scancode_required_phrases.inference import words_from_text
from scancode_required_phrases.model_rules import file_sha256
from scancode_required_phrases.model_rules import prepare_predicted_phrases
from scancode_required_phrases.model_rules import serialize_rule
from scancode_required_phrases.model_rules import write_rule_atomically


FORMAT_VERSION = 1
PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
DECISIONS = {PENDING, APPROVED, REJECTED}
DECISION_SOURCES = {None, "human", "score"}
VALIDATION_ISSUES = {None, "rejected", "not_found", "ambiguous"}
TARGET_MODES = {"rule", "rules_dir", "all"}
RUN_MODES = {"interactive", "batch"}

METADATA_FIELDS = {
    "record_type",
    "format_version",
    "model",
    "model_revision",
    "target_mode",
    "target",
    "run_mode",
    "auto_score",
    "review_score",
    "rules_scanned",
    "rules_eligible",
    "truncated_rules",
}
RULE_FIELDS = {
    "record_type",
    "path",
    "identifier",
    "license_expression",
    "original_hash",
    "truncated",
    "predictions",
    "expected_hash",
    "applied_hash",
}
PREDICTION_FIELDS = {
    "predicted_text",
    "text",
    "start_word",
    "end_word",
    "score",
    "validation_issue",
    "decision",
    "decision_source",
}


def _is_hash(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float))


def create_metadata(
    model,
    model_revision,
    target_mode,
    target,
    run_mode,
    rules_scanned,
    rules_eligible,
    truncated_rules,
    auto_score=None,
    review_score=None,
):
    """Return metadata for a new review session."""
    return {
        "record_type": "metadata",
        "format_version": FORMAT_VERSION,
        "model": model,
        "model_revision": model_revision,
        "target_mode": target_mode,
        "target": str(target),
        "run_mode": run_mode,
        "auto_score": auto_score,
        "review_score": review_score,
        "rules_scanned": rules_scanned,
        "rules_eligible": rules_eligible,
        "truncated_rules": truncated_rules,
    }


def create_prediction(prediction, validation_issue=None):
    """Return a pending session entry for one model prediction."""
    return {
        "predicted_text": prediction.text,
        "text": prediction.text,
        "start_word": prediction.start_word,
        "end_word": prediction.end_word,
        "score": prediction.score,
        "validation_issue": validation_issue,
        "decision": PENDING,
        "decision_source": None,
    }


def create_rule_record(rule_path, rule, predictions, truncated):
    """Return a session entry for one rule with predictions."""
    return {
        "record_type": "rule",
        "path": str(rule_path),
        "identifier": rule.identifier,
        "license_expression": rule.license_expression,
        "original_hash": file_sha256(rule_path),
        "truncated": truncated,
        "predictions": predictions,
        "expected_hash": None,
        "applied_hash": None,
    }


def _validate_metadata(metadata, session_path):
    location = f"{session_path} metadata"
    if type(metadata) is not dict or set(metadata) != METADATA_FIELDS:
        raise ValueError(f"{location}: invalid fields")
    if metadata["record_type"] != "metadata":
        raise ValueError(f"{location}: invalid record type")
    if metadata["format_version"] != FORMAT_VERSION:
        raise ValueError(f"{location}: unsupported format version {metadata['format_version']!r}")
    if type(metadata["model"]) is not str or not metadata["model"]:
        raise ValueError(f"{location}: model must be a non-empty string")

    revision = metadata["model_revision"]
    if revision is not None and (
        type(revision) is not str
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError(f"{location}: model revision is invalid")
    if metadata["target_mode"] not in TARGET_MODES:
        raise ValueError(f"{location}: target mode is invalid")
    if metadata["run_mode"] not in RUN_MODES:
        raise ValueError(f"{location}: run mode is invalid")

    target = metadata["target"]
    if type(target) is not str or not target or not Path(target).is_absolute():
        raise ValueError(f"{location}: target must be an absolute path")

    for field in ("rules_scanned", "rules_eligible", "truncated_rules"):
        value = metadata[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{location}: {field} must be a non-negative integer")
    if metadata["rules_eligible"] > metadata["rules_scanned"]:
        raise ValueError(f"{location}: eligible rule count exceeds scanned rules")
    if metadata["truncated_rules"] > metadata["rules_eligible"]:
        raise ValueError(f"{location}: truncated rule count exceeds eligible rules")

    auto_score = metadata["auto_score"]
    review_score = metadata["review_score"]
    if metadata["run_mode"] == "interactive":
        if auto_score is not None or review_score is not None:
            raise ValueError(f"{location}: interactive sessions cannot have thresholds")
    else:
        if not _is_number(auto_score) or not _is_number(review_score):
            raise ValueError(f"{location}: batch thresholds must be numbers")
        if not (
            math.isfinite(auto_score)
            and math.isfinite(review_score)
            and 0 <= review_score <= auto_score <= 1
        ):
            raise ValueError(f"{location}: batch thresholds are invalid")


def _validate_prediction(prediction, metadata, truncated, location):
    if type(prediction) is not dict or set(prediction) != PREDICTION_FIELDS:
        raise ValueError(f"{location}: invalid fields")

    for field in ("predicted_text", "text"):
        if type(prediction[field]) is not str or not prediction[field]:
            raise ValueError(f"{location}: {field} must be a non-empty string")
    for field in ("start_word", "end_word"):
        value = prediction[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{location}: {field} must be an integer")
    if prediction["start_word"] < 0 or prediction["end_word"] < prediction["start_word"]:
        raise ValueError(f"{location}: word offsets are invalid")

    score = prediction["score"]
    if not _is_number(score) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{location}: score must be between 0 and 1")
    issue = prediction["validation_issue"]
    decision = prediction["decision"]
    source = prediction["decision_source"]
    if issue not in VALIDATION_ISSUES:
        raise ValueError(f"{location}: validation issue is invalid")
    if decision not in DECISIONS:
        raise ValueError(f"{location}: decision is invalid")
    if source not in DECISION_SOURCES:
        raise ValueError(f"{location}: decision source is invalid")

    if issue and decision != PENDING:
        raise ValueError(f"{location}: invalid predictions cannot have decisions")
    if decision == PENDING and source is not None:
        raise ValueError(f"{location}: pending prediction cannot have a source")
    if decision == APPROVED and source not in {"human", "score"}:
        raise ValueError(f"{location}: approved prediction has no decision source")
    if decision == REJECTED and source != "human":
        raise ValueError(f"{location}: rejected prediction must be a human decision")
    if (
        metadata["run_mode"] == "batch"
        and issue is None
        and score < metadata["review_score"]
        and decision != PENDING
    ):
        raise ValueError(f"{location}: below-threshold prediction cannot have a decision")
    if source == "score":
        if metadata["run_mode"] != "batch":
            raise ValueError(f"{location}: score decision requires batch mode")
        if score < metadata["auto_score"]:
            raise ValueError(f"{location}: score decision is below the automatic threshold")
        if truncated:
            raise ValueError(f"{location}: truncated prediction cannot be score-approved")
        if prediction["text"] != prediction["predicted_text"]:
            raise ValueError(f"{location}: score-approved prediction cannot be edited")


def _validate_rule(record, metadata, session_path, line_number):
    location = f"{session_path} line {line_number}"
    if type(record) is not dict or set(record) != RULE_FIELDS:
        raise ValueError(f"{location}: invalid fields")
    if record["record_type"] != "rule":
        raise ValueError(f"{location}: invalid record type")

    path = record["path"]
    identifier = record["identifier"]
    expression = record["license_expression"]
    if type(path) is not str or not Path(path).is_absolute():
        raise ValueError(f"{location}: path must be absolute")
    if type(identifier) is not str or Path(path).name != identifier:
        raise ValueError(f"{location}: identifier does not match path")
    if not identifier.endswith(".RULE"):
        raise ValueError(f"{location}: identifier must end with .RULE")
    if type(expression) is not str or not expression:
        raise ValueError(f"{location}: license expression must be a non-empty string")
    if not _is_hash(record["original_hash"]):
        raise ValueError(f"{location}: original hash is invalid")
    if type(record["truncated"]) is not bool:
        raise ValueError(f"{location}: truncated must be a boolean")

    expected_hash = record["expected_hash"]
    applied_hash = record["applied_hash"]
    if expected_hash is not None and not _is_hash(expected_hash):
        raise ValueError(f"{location}: expected hash is invalid")
    if applied_hash is not None and not _is_hash(applied_hash):
        raise ValueError(f"{location}: applied hash is invalid")
    if expected_hash == record["original_hash"]:
        raise ValueError(f"{location}: expected hash equals original hash")
    if applied_hash is not None and applied_hash != expected_hash:
        raise ValueError(f"{location}: applied hash does not match expected hash")

    predictions = record["predictions"]
    if type(predictions) is not list or not predictions:
        raise ValueError(f"{location}: predictions must be a non-empty list")
    seen = set()
    for prediction_number, prediction in enumerate(predictions, 1):
        prediction_location = f"{location}, prediction {prediction_number}"
        _validate_prediction(
            prediction,
            metadata=metadata,
            truncated=record["truncated"],
            location=prediction_location,
        )
        identity = (
            prediction["predicted_text"],
            prediction["start_word"],
            prediction["end_word"],
        )
        if identity in seen:
            raise ValueError(f"{location}: duplicate prediction")
        seen.add(identity)

    has_pending = any(prediction_needs_review(metadata, prediction) for prediction in predictions)
    has_approved = any(
        prediction["decision"] == APPROVED
        and prediction["validation_issue"] is None
        and not prediction_is_below_threshold(metadata, prediction)
        for prediction in predictions
    )
    if expected_hash is not None and has_pending:
        raise ValueError(f"{location}: prepared rule has pending predictions")
    if expected_hash is not None and not has_approved:
        raise ValueError(f"{location}: prepared rule has no approved predictions")


def validate_session(metadata, records, session_path):
    """Validate complete session data without opening any rule path."""
    _validate_metadata(metadata, session_path)
    if type(records) is not list:
        raise ValueError(f"{session_path}: rule records must be a list")

    paths = set()
    identifiers = set()
    for line_number, record in enumerate(records, 2):
        _validate_rule(record, metadata, session_path, line_number)
        if record["path"] in paths:
            raise ValueError(f"{session_path} line {line_number}: duplicate rule path")
        if record["identifier"] in identifiers:
            raise ValueError(f"{session_path} line {line_number}: duplicate rule identifier")
        paths.add(record["path"])
        identifiers.add(record["identifier"])


def create_session_path(session_path=None):
    """Return a new explicit or automatically reserved session path."""
    if session_path:
        session_path = Path(session_path)
        if session_path.exists() or session_path.is_symlink():
            raise ValueError(f"Session file already exists: {session_path}")
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path = session_path.parent.resolve() / session_path.name
        descriptor = os.open(
            session_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
        os.close(descriptor)
        return session_path

    sessions_directory = (
        Path(click.get_app_dir("scancode-required-phrases", roaming=False)) / "sessions"
    )
    sessions_directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix="model-required-phrases-",
        suffix=".jsonl",
        dir=sessions_directory,
    )
    os.close(descriptor)
    return Path(name).resolve()


def write_session(session_path, metadata, records):
    """Validate and atomically replace a JSONL session."""
    session_path = Path(session_path)
    validate_session(metadata, records, session_path)
    if session_path.is_symlink():
        raise ValueError(f"Session file cannot be a symbolic link: {session_path}")
    if not session_path.parent.is_dir():
        raise ValueError(f"Session directory does not exist: {session_path.parent}")

    descriptor, name = tempfile.mkstemp(
        prefix=f".{session_path.name}.",
        suffix=".tmp",
        dir=session_path.parent,
    )
    temporary_path = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(metadata, ensure_ascii=False, allow_nan=False) + "\n")
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        if session_path.is_symlink():
            raise ValueError(f"Session file cannot be a symbolic link: {session_path}")
        os.replace(temporary_path, session_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def read_session(session_path):
    """Return validated metadata and records from a JSONL session."""
    session_path = Path(session_path)
    if session_path.is_symlink():
        raise ValueError(f"Session file cannot be a symbolic link: {session_path}")
    try:
        lines = session_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Cannot read session {session_path}: {error}") from error
    if not lines:
        raise ValueError(f"Session is empty: {session_path}")

    entries = []
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ValueError(f"{session_path} line {line_number}: empty line")
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{session_path} line {line_number}: malformed JSON: {error.msg}"
            ) from error

    metadata, records = entries[0], entries[1:]
    validate_session(metadata, records, session_path)
    return metadata, records


def validate_session_paths(metadata, records):
    """Validate every rule path against the recorded target."""
    target = Path(metadata["target"])
    if target.is_symlink():
        raise ValueError(f"Session target cannot be a symbolic link: {target}")
    resolved_target = target.resolve(strict=False)
    if resolved_target != target:
        raise ValueError(f"Session target is not canonical: {target}")

    if metadata["target_mode"] == "all":
        installed_rules = Path(rules_data_dir).resolve(strict=True)
        if target != installed_rules:
            raise ValueError("Session target is not the installed ScanCode rules directory")

    if metadata["target_mode"] == "rule" and len(records) > 1:
        raise ValueError("Single-rule session contains multiple rule records")

    for record in records:
        rule_path = Path(record["path"])
        if rule_path.is_symlink():
            raise ValueError(f"Rule file cannot be a symbolic link: {rule_path}")
        resolved_path = rule_path.resolve(strict=False)
        if resolved_path != rule_path:
            raise ValueError(f"Rule path is not canonical: {rule_path}")
        if metadata["target_mode"] == "rule":
            if rule_path != target:
                raise ValueError(f"Rule path does not match the session target: {rule_path}")
        elif not rule_path.is_relative_to(target):
            raise ValueError(f"Rule path is outside the session target: {rule_path}")


def prediction_is_below_threshold(metadata, prediction):
    """Return whether a valid batch prediction is below the review threshold."""
    return (
        metadata["run_mode"] == "batch"
        and prediction["validation_issue"] is None
        and prediction["score"] < metadata["review_score"]
    )


def prediction_needs_review(metadata, prediction):
    """Return whether a prediction still needs a human decision."""
    return (
        prediction["validation_issue"] is None
        and not prediction_is_below_threshold(metadata, prediction)
        and prediction["decision"] == PENDING
    )


def _load_rule(record):
    """Load an unchanged rule and validate its recorded predictions."""
    rule_path = Path(record["path"])
    if not rule_path.is_file():
        raise ValueError(f"Rule file is missing: {rule_path}")
    if file_sha256(rule_path) != record["original_hash"]:
        raise ValueError(f"Rule file is stale: {rule_path}")

    rule = Rule.from_file(str(rule_path))
    if rule.identifier != record["identifier"]:
        raise ValueError(f"Rule identifier changed: {rule_path}")
    if rule.license_expression != record["license_expression"]:
        raise ValueError(f"Rule license expression changed: {rule_path}")

    words = words_from_text(rule.text)
    for prediction in record["predictions"]:
        start = prediction["start_word"]
        end = prediction["end_word"]
        if end >= len(words) or " ".join(words[start : end + 1]) != prediction["predicted_text"]:
            raise ValueError(f"Prediction offsets do not match rule text: {rule_path}")
    return rule


def load_session_rules(metadata, records):
    """Reconcile applied rules and load every remaining unchanged rule."""
    validate_session(metadata, records, "session")
    validate_session_paths(metadata, records)
    loaded_rules = {}
    reconciled = []

    for record in records:
        rule_path = Path(record["path"])
        if not rule_path.is_file():
            raise ValueError(f"Rule file is missing: {rule_path}")
        current_hash = file_sha256(rule_path)
        original_hash = record["original_hash"]
        expected_hash = record["expected_hash"]
        applied_hash = record["applied_hash"]

        if applied_hash is not None:
            if current_hash == applied_hash:
                continue
            if current_hash == original_hash:
                raise ValueError(f"Applied rule was reverted: {rule_path}")
            raise ValueError(f"Applied rule is stale: {rule_path}")

        if expected_hash is not None and current_hash == expected_hash:
            record["applied_hash"] = expected_hash
            reconciled.append(record["identifier"])
            continue
        if current_hash != original_hash:
            raise ValueError(f"Rule file is stale: {rule_path}")
        loaded_rules[record["path"]] = _load_rule(record)

    return loaded_rules, reconciled


def prepare_rule_updates(metadata, records, loaded_rules):
    """Prepare fully decided rule updates before any write."""
    work = []
    unchanged = []
    deferred = []
    for record in records:
        if record["applied_hash"] is not None:
            continue
        if any(
            prediction_needs_review(metadata, prediction) for prediction in record["predictions"]
        ):
            record["expected_hash"] = None
            deferred.append(record["identifier"])
            continue

        phrases = [
            prediction["text"]
            for prediction in record["predictions"]
            if prediction["decision"] == APPROVED
            and prediction["validation_issue"] is None
            and not prediction_is_below_threshold(metadata, prediction)
        ]
        if not phrases:
            record["expected_hash"] = None
            continue

        rule_path = Path(record["path"])
        rule = loaded_rules[record["path"]]
        updated_rule = prepare_predicted_phrases(rule, phrases)
        content = serialize_rule(updated_rule, rule_path)
        original_content = rule_path.read_bytes()
        if hashlib.sha256(original_content).hexdigest() != record["original_hash"]:
            raise ValueError(f"Rule file is stale: {rule_path}")
        if content == original_content:
            record["expected_hash"] = None
            unchanged.append(record["identifier"])
            continue

        record["expected_hash"] = hashlib.sha256(content).hexdigest()
        work.append((record, rule_path, content))
    return work, unchanged, deferred


def write_rule_updates(session_path, metadata, records, work):
    """Write prepared rules and save progress after each successful write."""
    write_session(session_path, metadata, records)
    for record, rule_path, content in work:
        written_hash = write_rule_atomically(
            rule_path=rule_path,
            content=content,
            expected_sha256=record["original_hash"],
        )
        record["applied_hash"] = written_hash
        write_session(session_path, metadata, records)
