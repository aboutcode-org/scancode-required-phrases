# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare model-predicted required phrases for ScanCode license rules."""

from copy import copy
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import stat
import tempfile

from licensedcode.cache import get_licenses_db
from licensedcode.models import Rule
from licensedcode.models import rules_data_dir
from licensedcode.models import validate_rules
from licensedcode.required_phrases import add_required_phrase_to_rule
from licensedcode.required_phrases import find_phrase_spans_in_text
from licensedcode.required_phrases import get_updatable_rules_by_expression
from licensedcode.required_phrases import RequiredPhraseRuleCandidate
from licensedcode.tokenize import get_existing_required_phrase_spans

from scancode_required_phrases.inference import RequiredPhrasePredictor
from scancode_required_phrases.training import IMMUTABLE_REVISION


MIN_TOKENS = 2
MIN_SINGLE_TOKEN_LEN = 5
MAX_RULE_LENGTH = 4000


def _artifact_names(success_marker):
    """Return safe artifact paths listed in a model success marker."""
    try:
        files = success_marker["files"]
    except (KeyError, TypeError) as error:
        raise ValueError("Remote model has no valid file inventory") from error
    if type(files) is not dict or not files:
        raise ValueError("Remote model has no valid file inventory")

    names = ["SUCCESS.json", *files]
    for name in names:
        if type(name) is not str or not name or "\\" in name:
            raise ValueError(f"Remote model contains an unsafe artifact path: {name!r}")
        path = PurePosixPath(name)
        if (
            not path.parts
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != name
        ):
            raise ValueError(f"Remote model contains an unsafe artifact path: {name!r}")
    return names


def _load_remote_predictor(repository, revision, hf_token):
    """Download only declared model artifacts and load them locally."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub import snapshot_download

    marker_path = hf_hub_download(
        repo_id=repository,
        filename="SUCCESS.json",
        revision=revision,
        token=hf_token,
    )
    try:
        marker = json.loads(Path(marker_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Remote model has no valid success marker") from error

    artifact_names = _artifact_names(marker)
    snapshot = Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            token=hf_token,
            allow_patterns=artifact_names,
        )
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_dir = Path(temporary_directory)
        for name in artifact_names:
            source = snapshot / name
            if not source.is_file():
                raise ValueError(f"Remote model is missing artifact: {name}")
            source = source.resolve(strict=True)
            target = model_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.hardlink_to(source)
            except OSError:
                shutil.copy2(source, target)
        return RequiredPhrasePredictor.from_model_dir(model_dir)


def load_predictor(model, hf_token=None, revision=None):
    """Load a predictor from a local directory or pinned Hugging Face revision."""
    model_dir = Path(model)
    if model_dir.is_dir():
        if revision:
            raise ValueError("A model revision cannot be used with a local model directory")
        return RequiredPhrasePredictor.from_model_dir(model_dir)

    if not revision or not IMMUTABLE_REVISION.fullmatch(revision):
        raise ValueError("A 40-character model revision is required for a remote model")
    return _load_remote_predictor(model, revision, hf_token)


def prediction_rule_issue(rule):
    """Return why a rule is not eligible for prediction, or None."""
    if rule.is_deprecated:
        return "deprecated"
    if rule.is_from_license:
        return "from_license"
    if len(rule.text) > MAX_RULE_LENGTH:
        return "too_long"
    if not rule.is_approx_matchable:
        return "not_approx_matchable"
    if rule.skip_for_required_phrase_generation:
        return "skipped"
    if get_existing_required_phrase_spans(rule.text):
        return "has_required_phrases"


def _load_rule_file(rule_path):
    """Return a resolved rule path and its loaded rule."""
    rule_path = Path(rule_path)
    if rule_path.is_symlink():
        raise ValueError(f"Rule file cannot be a symbolic link: {rule_path}")
    if not rule_path.exists():
        raise ValueError(f"Rule file does not exist: {rule_path}")
    if not rule_path.is_file():
        raise ValueError(f"Rule path is not a file: {rule_path}")
    if rule_path.suffix != ".RULE":
        raise ValueError(f"Rule file must end with .RULE: {rule_path}")

    rule_path = rule_path.resolve(strict=True)
    return rule_path, Rule.from_file(str(rule_path))


def load_prediction_rule(rule_path, license_expression=None):
    """Load and validate one eligible rule and return its resolved path."""
    rule_path, rule = _load_rule_file(rule_path)
    validate_rules(
        rules=[rule],
        licenses_by_key=get_licenses_db(),
    )

    if license_expression and rule.license_expression != license_expression:
        raise ValueError(
            f"Rule {rule.identifier} does not use expression: {license_expression}"
        )
    issue = prediction_rule_issue(rule)
    if issue:
        raise ValueError(f"Rule {rule.identifier} is not eligible: {issue}")
    return rule_path, rule


def load_prediction_rules(rules_directory, license_expression=None):
    """Load eligible top-level rules from a directory in filename order."""
    rules_directory = Path(rules_directory)
    if not rules_directory.exists():
        raise ValueError(f"Rules directory does not exist: {rules_directory}")
    if not rules_directory.is_dir():
        raise ValueError(f"Rules path is not a directory: {rules_directory}")
    rules_directory = rules_directory.resolve(strict=True)

    loaded = []
    for rule_path in sorted(rules_directory.glob("*.RULE")):
        resolved_path, rule = _load_rule_file(rule_path)
        if not resolved_path.is_relative_to(rules_directory):
            raise ValueError(f"Rule path is outside the rules directory: {rule_path}")
        loaded.append((resolved_path, rule))

    identifiers = [rule.identifier for _path, rule in loaded]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Rules directory contains duplicate identifiers: {rules_directory}")

    validate_rules(
        rules=[rule for _path, rule in loaded],
        licenses_by_key=get_licenses_db(),
    )
    return [
        (rule_path, rule)
        for rule_path, rule in loaded
        if (not license_expression or rule.license_expression == license_expression)
        and not prediction_rule_issue(rule)
    ]


def select_installed_prediction_rules(license_expression=None):
    """Return eligible installed rules and their resolved paths."""
    try:
        rules_by_expression = get_updatable_rules_by_expression(
            license_expression=license_expression,
            simple_expression=False,
        )
    except KeyError:
        raise ValueError(
            f"No rules for license expression: {license_expression}"
        ) from None

    rules_directory = Path(rules_data_dir).resolve(strict=True)
    selected = []
    for expression in sorted(rules_by_expression):
        for rule in sorted(rules_by_expression[expression], key=lambda item: item.identifier):
            if prediction_rule_issue(rule):
                continue
            rule_path = rules_directory / rule.identifier
            if rule_path.is_symlink():
                raise ValueError(f"Rule file cannot be a symbolic link: {rule_path}")
            rule_path = rule_path.resolve(strict=True)
            if not rule_path.is_relative_to(rules_directory):
                raise ValueError(f"Rule path is outside the installed rules directory: {rule_path}")
            selected.append((rule_path, rule))
    return selected


def candidate_issue(rule, phrase):
    """Return why ``phrase`` cannot be safely added, or None."""
    candidate = RequiredPhraseRuleCandidate.create(rule.license_expression, phrase)
    if not candidate.is_good(rule, MIN_TOKENS, MIN_SINGLE_TOKEN_LEN):
        return "rejected"

    spans = find_phrase_spans_in_text(rule.text, phrase)
    if not spans:
        return "not_found"
    if len(spans) != 1:
        return "ambiguous"


def predict_rule_candidates(rule, predictor):
    """Return read-only model predictions and their validation issues."""
    result = predictor.predict(rule.text)
    candidates = tuple(
        (prediction, candidate_issue(rule, prediction.text))
        for prediction in result.phrases
    )
    return result, candidates


def prepare_predicted_phrases(rule, phrases):
    """Return a copy with every phrase added, or raise if the update is unsafe."""
    phrases = list(phrases)
    if not phrases:
        raise ValueError("No predicted phrases to add")
    if len(phrases) != len(set(phrases)):
        raise ValueError("Duplicate predicted phrase")

    for phrase in phrases:
        issue = candidate_issue(rule, phrase)
        if issue:
            raise ValueError(f"Phrase {phrase!r} is not safe to add: {issue}")

    updated_rule = copy(rule)
    source = rule.source or ""
    if "ml_model" not in source.split():
        source = f"{source} ml_model" if source else "ml_model"
    for phrase in sorted(phrases, key=lambda text: (-len(text), text)):
        if not add_required_phrase_to_rule(
            rule=updated_rule,
            required_phrase=phrase,
            source=source,
            dry_run=True,
        ):
            raise ValueError(f"Predicted phrases conflict at: {phrase!r}")
    return updated_rule


def serialize_rule(rule, rule_path):
    """Return the bytes ScanCode would write for a rule."""
    rule_path = Path(rule_path)
    if rule.identifier != rule_path.name:
        raise ValueError(
            f"Rule identifier {rule.identifier!r} does not match path {rule_path}"
        )

    with tempfile.TemporaryDirectory() as temporary_directory:
        rule.dump(temporary_directory)
        staged = Path(temporary_directory) / rule.identifier
        if not staged.is_file():
            raise ValueError(f"Rule cannot be serialized: {rule.identifier}")
        content = staged.read_bytes().replace(b"\r\n", b"\n")
        if b"\r\n" in rule_path.read_bytes():
            content = content.replace(b"\n", b"\r\n")
        return content


def file_sha256(rule_path):
    """Return the SHA-256 digest of a rule file."""
    return hashlib.sha256(Path(rule_path).read_bytes()).hexdigest()


def write_rule_atomically(rule_path, content, expected_sha256):
    """Atomically replace an unchanged rule file with precomputed content."""
    rule_path = Path(rule_path)
    if rule_path.is_symlink() or not rule_path.is_file():
        raise ValueError(f"Rule path is not a regular file: {rule_path}")

    file_mode = stat.S_IMODE(rule_path.stat().st_mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{rule_path.name}.",
        suffix=".tmp",
        dir=rule_path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary_path, file_mode)

        if file_sha256(rule_path) != expected_sha256:
            raise ValueError(f"Rule changed before it could be written: {rule_path}")
        os.replace(temporary_path, rule_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return hashlib.sha256(content).hexdigest()
