# -*- coding: utf-8 -*-
#
# Copyright (c) nexB Inc. and others. All rights reserved.
# ScanCode is a trademark of nexB Inc.
# SPDX-License-Identifier: Apache-2.0
# See http://www.apache.org/licenses/LICENSE-2.0 for the license text.
# See https://github.com/aboutcode-org/scancode-required-phrases for support or download.
# See https://aboutcode.org for more information about nexB OSS projects.
#

"""Add existing required phrases to composite ScanCode license rules."""

import click
from license_expression import Licensing

from licensedcode.cache import get_licenses_db
from licensedcode.models import rules_data_dir
from licensedcode.required_phrases import add_required_phrase_to_rule
from licensedcode.required_phrases import collect_is_required_phrase_from_rules
from licensedcode.required_phrases import find_phrase_spans_in_text
from licensedcode.required_phrases import get_base_rules_by_expression
from licensedcode.required_phrases import get_ignorable_spans
from licensedcode.required_phrases import get_non_overlapping_spans
from licensedcode.required_phrases import get_updatable_rules_by_expression
from licensedcode.required_phrases import validate_and_reindex
from licensedcode.tokenize import get_existing_required_phrase_spans


def get_required_phrases_by_key(rules_by_expression, licenses_by_key):
    """
    Return required phrase candidates grouped by license key.
    Only collect from required phrase rules with a single non-generic key.
    """
    licensing = Licensing()
    required_phrases_by_key = {}
    required_phrases_by_expression = collect_is_required_phrase_from_rules(
        rules_by_expression=rules_by_expression,
    )

    for expression, required_phrases in required_phrases_by_expression.items():
        license_keys = licensing.license_keys(expression, unique=True)
        if len(license_keys) != 1:
            continue

        license_key = license_keys[0]
        if licenses_by_key[license_key].is_generic:
            continue

        if required_phrases:
            required_phrases_by_key[license_key] = required_phrases

    return required_phrases_by_key


def _get_required_phrase_matches(rule, license_keys, required_phrases_by_key):
    """
    Return one non-overlapping required phrase match for every license key.
    Prefer phrases already marked in the rule and return None if no complete match exists.
    """
    existing_spans = get_existing_required_phrase_spans(rule.text)
    unavailable_spans = existing_spans + get_ignorable_spans(rule)
    matches_by_key = {}

    for license_key in license_keys:
        marked_matches = []
        new_matches = []

        for candidate in required_phrases_by_key.get(license_key, []):
            phrase_spans = find_phrase_spans_in_text(
                rule.text,
                candidate.required_phrase_text,
            )
            marked_spans = [
                span
                for span in phrase_spans
                if any(span in existing for existing in existing_spans)
            ]
            if marked_spans:
                marked_matches.extend((candidate, True, [span]) for span in marked_spans)
                continue

            spans_to_add = list(
                get_non_overlapping_spans(
                    old_required_phrase_spans=unavailable_spans,
                    new_required_phrase_spans=phrase_spans,
                )
            )
            if spans_to_add:
                new_matches.append((candidate, False, spans_to_add))

        matches_by_key[license_key] = marked_matches + new_matches
        if not matches_by_key[license_key]:
            return

    def find_matches(remaining_keys, matched_spans):
        if not remaining_keys:
            return []

        license_key = remaining_keys[0]
        for required_phrase, is_marked, phrase_spans in matches_by_key[license_key]:
            if any(span.overlap(matched) for span in phrase_spans for matched in matched_spans):
                continue

            remaining_matches = find_matches(
                remaining_keys=remaining_keys[1:],
                matched_spans=matched_spans + phrase_spans,
            )
            if remaining_matches is not None:
                return [
                    (required_phrase, is_marked),
                    *remaining_matches,
                ]

    return find_matches(
        remaining_keys=license_keys,
        matched_spans=[],
    )


def add_required_phrases_to_composite_rules(
    rules,
    license_keys,
    required_phrases_by_key,
    write_phrase_source=False,
    dry_run=False,
):
    """Add required phrases to rules when every license key has a matching phrase."""
    for rule in rules:
        matched_required_phrases = _get_required_phrase_matches(
            rule=rule,
            license_keys=license_keys,
            required_phrases_by_key=required_phrases_by_key,
        )
        if not matched_required_phrases:
            continue

        original_text = rule.text
        original_source = rule.source
        updated = False

        for required_phrase, is_marked in matched_required_phrases:
            if is_marked:
                continue

            source = rule.source
            if write_phrase_source:
                phrase_source = required_phrase.rule.identifier
                source = f"{source} {phrase_source}" if source else phrase_source

            added = add_required_phrase_to_rule(
                rule=rule,
                required_phrase=required_phrase.required_phrase_text,
                source=source,
                dry_run=True,
            )
            if not added:
                rule.text = original_text
                rule.source = original_source
                updated = False
                break

            updated = True

        if updated and not dry_run:
            rule.dump(rules_data_dir)


def update_composite_rules_using_required_phrases(
    license_expression=None,
    write_phrase_source=False,
    verbose=False,
    dry_run=False,
):
    """
    Add existing required phrases to composite rules when every non-generic
    license key has a non-overlapping match.
    """
    licensing = Licensing()
    licenses_by_key = get_licenses_db()
    rules_by_expression = get_base_rules_by_expression()
    required_phrases_by_key = get_required_phrases_by_key(
        rules_by_expression=rules_by_expression,
        licenses_by_key=licenses_by_key,
    )
    updatable_rules_by_expression = get_updatable_rules_by_expression(
        license_expression=license_expression,
        simple_expression=False,
    )

    for expression, updatable_rules in updatable_rules_by_expression.items():
        license_keys = licensing.license_keys(expression, unique=True)
        if len(license_keys) < 2:
            continue

        license_keys = [
            license_key
            for license_key in license_keys
            if not licenses_by_key[license_key].is_generic
        ]
        if not license_keys:
            continue

        if verbose:
            click.echo(f"Annotating required phrases for expression: {expression}")

        add_required_phrases_to_composite_rules(
            rules=updatable_rules,
            license_keys=license_keys,
            required_phrases_by_key=required_phrases_by_key,
            write_phrase_source=write_phrase_source,
            dry_run=dry_run,
        )


@click.command(name="add-composite-required-phrases")
@click.option(
    "-l",
    "--license-expression",
    type=str,
    default=None,
    metavar="STRING",
    help="Only update rules using this license expression.",
)
@click.option(
    "-w",
    "--write-phrase-source",
    is_flag=True,
    default=False,
    help="Record the source required phrase rule in modified rules.",
)
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="Validate all rules and licenses after updating.",
)
@click.option(
    "-r",
    "--reindex",
    is_flag=True,
    default=False,
    help="Rebuild and cache the license index after updating.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Do not save rules.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Print progress information.",
)
@click.help_option("-h", "--help")
def add_composite_required_phrases(
    license_expression,
    write_phrase_source,
    validate,
    reindex,
    dry_run,
    verbose,
):
    """Add existing required phrases to composite license rules."""
    click.echo("Updating composite rules from required phrases.")
    update_composite_rules_using_required_phrases(
        license_expression=license_expression,
        write_phrase_source=write_phrase_source,
        dry_run=dry_run,
        verbose=verbose,
    )
    validate_and_reindex(
        validate=validate,
        reindex=reindex,
        verbose=verbose,
    )
