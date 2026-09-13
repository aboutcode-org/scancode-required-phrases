# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add model-predicted required phrases to ScanCode license rules."""

from copy import copy
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import tempfile

import click

from licensedcode.models import rules_data_dir
from licensedcode.required_phrases import add_required_phrase_to_rule
from licensedcode.required_phrases import find_phrase_spans_in_text
from licensedcode.required_phrases import get_updatable_rules_by_expression
from licensedcode.required_phrases import RequiredPhraseRuleCandidate
from licensedcode.tokenize import get_existing_required_phrase_spans

from scancode_required_phrases.inference import RequiredPhrasePredictor
from scancode_required_phrases.training import IMMUTABLE_REVISION


MIN_TOKENS = 2
MIN_SINGLE_TOKEN_LEN = 5


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
        if type(name) is not str or not name:
            raise ValueError(f"Remote model contains an unsafe artifact path: {name!r}")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
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


def select_rules(license_expression=None):
    """Return rules eligible for model prediction, grouped by expression."""
    try:
        rules_by_expression = get_updatable_rules_by_expression(
            license_expression=license_expression,
            simple_expression=False,
        )
    except KeyError:
        raise click.ClickException(
            f"No rules for license expression: {license_expression}"
        ) from None

    selected_rules_by_expression = {}
    for expression, rules in rules_by_expression.items():
        selected_rules = [rule for rule in rules if not get_existing_required_phrase_spans(rule.text)]
        if selected_rules:
            selected_rules_by_expression[expression] = selected_rules
    return selected_rules_by_expression


def new_counts():
    return dict(
        rules=0,
        truncated=0,
        rejected=0,
        not_found=0,
        ambiguous=0,
        conflicts=0,
        injected=0,
        changed=0,
        written=0,
    )


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


def add_predicted_phrases(rule, phrases, counts, dry_run=False, verbose=False):
    """Validate a complete rule update and write the rule at most once."""
    accepted_phrases = []
    for phrase in phrases:
        issue = candidate_issue(rule, phrase)
        if issue:
            counts[issue] += 1
            continue
        accepted_phrases.append(phrase)

    if not accepted_phrases:
        return False

    accepted_phrases.sort(key=lambda phrase: (-len(phrase), phrase))
    updated_rule = copy(rule)
    source = f"{rule.source} ml_model" if rule.source else "ml_model"
    for phrase in accepted_phrases:
        if not add_required_phrase_to_rule(
            rule=updated_rule,
            required_phrase=phrase,
            source=source,
            debug=verbose,
            dry_run=True,
        ):
            counts["conflicts"] += 1
            return False

    if updated_rule.text == rule.text:
        return False

    counts["injected"] += len(accepted_phrases)
    if dry_run:
        return True

    updated_rule.dump(rules_data_dir)
    rule.text = updated_rule.text
    rule.source = updated_rule.source
    return True


def update_rules_from_predictions(
    selected,
    predictor,
    dry_run=False,
    limit=0,
    verbose=False,
):
    """Predict and add phrases to selected rules and return run counts."""
    counts = new_counts()
    total = sum(len(rules) for rules in selected.values())
    click.echo(f"Predicting required phrases for {total} rules")

    for expression, rules in selected.items():
        if verbose:
            click.echo(f"{expression}: {len(rules)} rules")

        for rule in rules:
            if limit and counts["rules"] >= limit:
                click.echo(f"Stopping at {limit} rules")
                return counts

            counts["rules"] += 1
            result = predictor.predict(rule.text)
            if result.truncated:
                counts["truncated"] += 1
            phrases = [prediction.text for prediction in result.phrases]
            if not phrases:
                continue

            if verbose:
                click.echo(f"  {rule.identifier}: {phrases}")
            if add_predicted_phrases(
                rule=rule,
                phrases=phrases,
                counts=counts,
                dry_run=dry_run,
                verbose=verbose,
            ):
                counts["changed"] += 1
                if not dry_run:
                    counts["written"] += 1

    return counts


@click.command(name="add-model-required-phrases")
@click.option(
    "--model",
    required=True,
    help="Final model directory or Hugging Face repository.",
)
@click.option(
    "--model-revision",
    help="Full commit hash required for a Hugging Face model.",
)
@click.option(
    "--license-expression",
    help="Only update rules for this license expression.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Predict and validate phrases without saving rules.",
)
@click.option(
    "--limit",
    default=0,
    type=click.IntRange(min=0),
    help="Stop after this many rules; zero processes all rules.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print predictions for each rule.",
)
@click.help_option("-h", "--help")
def add_model_required_phrases(
    model,
    model_revision,
    license_expression,
    dry_run,
    limit,
    verbose,
):
    """Add model-predicted required phrases to license rules."""
    selected = select_rules(license_expression=license_expression)
    if not selected:
        click.echo("No eligible rules found")
        return

    try:
        predictor = load_predictor(
            model,
            hf_token=os.environ.get("HF_TOKEN"),
            revision=model_revision,
        )
    except ValueError as error:
        raise click.ClickException(str(error)) from error
    counts = update_rules_from_predictions(
        selected=selected,
        predictor=predictor,
        dry_run=dry_run,
        limit=limit,
        verbose=verbose,
    )

    click.echo(f"\nrules processed  : {counts['rules']}")
    click.echo(f"  truncated      : {counts['truncated']}")
    click.echo(f"phrases accepted : {counts['injected']}")
    click.echo(f"  rejected       : {counts['rejected']}")
    click.echo(f"  not found      : {counts['not_found']}")
    click.echo(f"  ambiguous      : {counts['ambiguous']}")
    click.echo(f"  conflicts      : {counts['conflicts']}")
    click.echo(f"rules changed    : {counts['changed']}")
    click.echo(f"rules written    : {counts['written']}")

    if dry_run:
        click.echo("Dry run: no rules were saved")
    elif counts["written"]:
        click.echo("Run scancode-reindex-licenses to use the new required phrases")


if __name__ == "__main__":
    add_model_required_phrases()
