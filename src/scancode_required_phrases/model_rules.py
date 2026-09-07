# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add model-predicted required phrases to ScanCode license rules."""

import os
from pathlib import Path

import click

from licensedcode.models import rules_data_dir
from licensedcode.required_phrases import add_required_phrase_to_rule
from licensedcode.required_phrases import find_phrase_spans_in_text
from licensedcode.required_phrases import get_base_rules_by_expression
from licensedcode.required_phrases import RequiredPhraseRuleCandidate
from licensedcode.tokenize import get_existing_required_phrase_spans

from scancode_required_phrases.inference import RequiredPhrasePredictor


MIN_TOKENS = 2
MIN_SINGLE_TOKEN_LEN = 5
MAX_RULE_TEXT = 4000


def load_predictor(model, hf_token=None):
    """Load a predictor from a local directory or Hugging Face repository."""
    model_dir = Path(model)
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download

        model_dir = Path(snapshot_download(repo_id=model, token=hf_token))
    return RequiredPhrasePredictor.from_model_dir(model_dir)


def is_updatable(rule):
    """Return True if a rule can receive predicted required phrases."""
    if rule.is_from_license:
        return False
    if len(rule.text) > MAX_RULE_TEXT:
        return False
    if not rule.is_approx_matchable:
        return False
    if rule.skip_for_required_phrase_generation:
        return False
    return not get_existing_required_phrase_spans(rule.text)


def select_rules(license_expression=None):
    """Return eligible rules grouped by license expression."""
    try:
        rules_by_expression = get_base_rules_by_expression(license_expression)
    except KeyError:
        raise click.ClickException(
            f"No rules for license expression: {license_expression}"
        ) from None

    selected = {}
    for expression, rules in rules_by_expression.items():
        updatable = [rule for rule in rules if is_updatable(rule)]
        if updatable:
            selected[expression] = updatable
    return selected


def new_counts():
    return dict(
        rules=0,
        truncated=0,
        rejected=0,
        not_found=0,
        injected=0,
        skipped=0,
        written=0,
    )


def add_predicted_phrases(rule, phrases, counts, dry_run=False, verbose=False):
    """Validate and add predicted phrases, writing the rule at most once."""
    candidates = []
    for phrase in phrases:
        candidate = RequiredPhraseRuleCandidate.create(rule.license_expression, phrase)
        if not candidate.is_good(rule, MIN_TOKENS, MIN_SINGLE_TOKEN_LEN):
            counts["rejected"] += 1
            continue
        if not find_phrase_spans_in_text(rule.text, phrase):
            counts["not_found"] += 1
            continue
        candidates.append(phrase)

    if not candidates:
        return False

    original_text = rule.text
    original_source = rule.source
    source = f"{original_source} ml_model" if original_source else "ml_model"

    for phrase in candidates:
        updated = add_required_phrase_to_rule(
            rule=rule,
            required_phrase=phrase,
            source=source,
            debug=verbose,
            dry_run=True,
        )
        if updated:
            counts["injected"] += 1
        else:
            counts["skipped"] += 1

    if rule.text == original_text:
        return False
    if not dry_run:
        rule.dump(rules_data_dir)
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
                counts["written"] += 1

    return counts


@click.command(name="add-model-required-phrases")
@click.option(
    "--model",
    required=True,
    help="Final model directory or Hugging Face repository.",
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
def add_model_required_phrases(model, license_expression, dry_run, limit, verbose):
    """Add model-predicted required phrases to license rules."""
    selected = select_rules(license_expression=license_expression)
    if not selected:
        click.echo("No eligible rules found")
        return

    predictor = load_predictor(model, hf_token=os.environ.get("HF_TOKEN"))
    counts = update_rules_from_predictions(
        selected=selected,
        predictor=predictor,
        dry_run=dry_run,
        limit=limit,
        verbose=verbose,
    )

    click.echo(f"\nrules processed  : {counts['rules']}")
    click.echo(f"  truncated      : {counts['truncated']}")
    click.echo(f"phrases injected : {counts['injected']}")
    click.echo(f"  rejected       : {counts['rejected']}")
    click.echo(f"  not found      : {counts['not_found']}")
    click.echo(f"  nothing to add : {counts['skipped']}")
    click.echo(f"rules written    : {counts['written']}")

    if dry_run:
        click.echo("Dry run: no rules were saved")
    elif counts["written"]:
        click.echo("Run scancode-reindex-licenses to use the new required phrases")


if __name__ == "__main__":
    add_model_required_phrases()
