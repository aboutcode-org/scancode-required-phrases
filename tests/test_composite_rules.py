# -*- coding: utf-8 -*-
#
# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from click.testing import CliRunner
import pytest

from licensedcode.models import Rule
from licensedcode.required_phrases import IsRequiredPhrase

from scancode_required_phrases import composite_rules


def make_required_phrase_rule(expression, text, identifier):
    return SimpleNamespace(
        license_expression=expression,
        text=text,
        identifier=identifier,
        is_required_phrase=True,
    )


def make_license(is_generic=False):
    return SimpleNamespace(is_generic=is_generic)


def make_candidate(expression, text, identifier):
    rule = make_required_phrase_rule(expression, text, identifier)
    return IsRequiredPhrase(rule=rule, required_phrase_text=text)


@pytest.fixture
def required_phrases_by_key():
    return {
        "mit": [make_candidate("mit", "MIT License", "mit_1.RULE")],
        "apache-2.0": [
            make_candidate("apache-2.0", "Apache License", "apache-2.0_1.RULE"),
        ],
        "bsd-new": [make_candidate("bsd-new", "BSD License", "bsd-new_1.RULE")],
    }


def test_get_required_phrases_by_key_collects_single_key_phrases_longest_first():
    rules_by_expression = {
        "mit": [
            make_required_phrase_rule("mit", "MIT", "mit_1.RULE"),
            make_required_phrase_rule("mit", "MIT License", "mit_2.RULE"),
            SimpleNamespace(is_required_phrase=False),
        ],
    }

    required_phrases = composite_rules.get_required_phrases_by_key(
        rules_by_expression=rules_by_expression,
        licenses_by_key={"mit": make_license()},
    )

    assert [phrase.required_phrase_text for phrase in required_phrases["mit"]] == [
        "MIT License",
        "MIT",
    ]


def test_get_required_phrases_by_key_skips_composite_expressions():
    rules_by_expression = {
        "mit AND apache-2.0": [
            make_required_phrase_rule(
                "mit AND apache-2.0",
                "MIT and Apache",
                "mit_and_apache_1.RULE",
            ),
        ],
    }
    licenses_by_key = {
        "mit": make_license(),
        "apache-2.0": make_license(),
    }

    required_phrases = composite_rules.get_required_phrases_by_key(
        rules_by_expression=rules_by_expression,
        licenses_by_key=licenses_by_key,
    )

    assert required_phrases == {}


def test_get_required_phrases_by_key_skips_generic_licenses():
    rules_by_expression = {
        "unknown": [
            make_required_phrase_rule("unknown", "Unknown License", "unknown_1.RULE"),
        ],
    }

    required_phrases = composite_rules.get_required_phrases_by_key(
        rules_by_expression=rules_by_expression,
        licenses_by_key={"unknown": make_license(is_generic=True)},
    )

    assert required_phrases == {}


def test_add_required_phrases_marks_every_key(required_phrases_by_key):
    rule = Rule(
        license_expression="mit AND apache-2.0",
        identifier="mit_and_apache_test.RULE",
        text="Licensed under the MIT License and the Apache License.",
        is_license_notice=True,
    )

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["mit", "apache-2.0"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert "{{MIT License}}" in rule.text
    assert "{{Apache License}}" in rule.text


def test_add_required_phrases_requires_every_key(required_phrases_by_key):
    text = "Licensed under the MIT License."
    rule = Rule(
        license_expression="mit AND apache-2.0",
        identifier="missing_apache_test.RULE",
        text=text,
        is_license_notice=True,
    )

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["mit", "apache-2.0"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert rule.text == text
    assert rule.source is None


def test_add_required_phrases_marks_three_keys(required_phrases_by_key):
    rule = Rule(
        license_expression="mit AND apache-2.0 AND bsd-new",
        identifier="three_key_test.RULE",
        text="MIT License, Apache License, and BSD License apply.",
        is_license_notice=True,
    )

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["mit", "apache-2.0", "bsd-new"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert "{{MIT License}}" in rule.text
    assert "{{Apache License}}" in rule.text
    assert "{{BSD License}}" in rule.text


def test_add_required_phrases_keeps_existing_markers(required_phrases_by_key):
    rule = Rule(
        license_expression="mit AND apache-2.0",
        identifier="existing_marker_test.RULE",
        text="Licensed under the {{MIT License}} and the Apache License.",
        is_license_notice=True,
    )

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["mit", "apache-2.0"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert rule.text.count("{{MIT License}}") == 1
    assert "{{Apache License}}" in rule.text


def test_add_required_phrases_prefers_an_existing_marker(required_phrases_by_key):
    text = "Licensed under {{Apache License}} {{or the MIT License}} (LICENSE.mit)."
    rule = Rule(
        license_expression="mit OR apache-2.0",
        identifier="existing_markers_test.RULE",
        text=text,
        is_license_notice=True,
    )
    required_phrases_by_key["mit"] = [
        make_candidate("mit", "License: MIT", "mit_2.RULE"),
        required_phrases_by_key["mit"][0],
    ]

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["mit", "apache-2.0"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert rule.text == text


def test_add_required_phrases_writes_once(required_phrases_by_key, tmp_path, monkeypatch):
    rule = Rule(
        license_expression="mit AND apache-2.0",
        identifier="write_once_test.RULE",
        text="Licensed under the MIT License and the Apache License.",
        is_license_notice=True,
    )
    original_dump = Rule.dump
    dump_calls = []

    def dump(rule, rules_data_dir):
        dump_calls.append(rule.identifier)
        original_dump(rule, rules_data_dir)

    monkeypatch.setattr(Rule, "dump", dump)
    monkeypatch.setattr(composite_rules, "rules_data_dir", str(tmp_path))

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["mit", "apache-2.0"],
        required_phrases_by_key=required_phrases_by_key,
        write_phrase_source=True,
    )

    saved_rule = Rule.from_file(str(tmp_path / rule.identifier))
    assert dump_calls == [rule.identifier]
    assert "{{MIT License}}" in saved_rule.text
    assert "{{Apache License}}" in saved_rule.text
    assert saved_rule.source == "mit_1.RULE apache-2.0_1.RULE"


def test_add_required_phrases_dry_run_does_not_write(
    required_phrases_by_key,
    monkeypatch,
):
    rule = Rule(
        license_expression="mit AND apache-2.0",
        identifier="dry_run_test.RULE",
        text="Licensed under the MIT License and the Apache License.",
        is_license_notice=True,
    )

    def dump(*args, **kwargs):
        pytest.fail("Rule.dump() called during a dry run")

    monkeypatch.setattr(Rule, "dump", dump)

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["mit", "apache-2.0"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert "{{MIT License}}" in rule.text
    assert "{{Apache License}}" in rule.text


def test_add_required_phrases_rolls_back_a_failed_update(
    required_phrases_by_key,
    monkeypatch,
):
    text = "Licensed under the MIT License and the Apache License."
    source = "existing.RULE"
    rule = Rule(
        license_expression="mit AND apache-2.0",
        identifier="rollback_test.RULE",
        text=text,
        source=source,
        is_license_notice=True,
    )
    original_add = composite_rules.add_required_phrase_to_rule
    calls = []

    def add_required_phrase(*args, **kwargs):
        calls.append(kwargs["required_phrase"])
        if len(calls) == 2:
            return False
        return original_add(*args, **kwargs)

    monkeypatch.setattr(composite_rules, "add_required_phrase_to_rule", add_required_phrase)

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["mit", "apache-2.0"],
        required_phrases_by_key=required_phrases_by_key,
        write_phrase_source=True,
        dry_run=True,
    )

    assert calls == ["MIT License", "Apache License"]
    assert rule.text == text
    assert rule.source == source


def test_add_required_phrases_uses_a_non_overlapping_candidate():
    rule = Rule(
        license_expression="gpl-2.0 AND gpl-2.0-plus",
        identifier="overlapping_candidate_test.RULE",
        text="GNU General Public License version 2, or any later version.",
        is_license_notice=True,
    )
    required_phrases_by_key = {
        "gpl-2.0": [
            make_candidate(
                "gpl-2.0",
                "GNU General Public License version 2",
                "gpl-2.0_1.RULE",
            ),
        ],
        "gpl-2.0-plus": [
            make_candidate(
                "gpl-2.0-plus",
                "General Public License version 2",
                "gpl-2.0-plus_1.RULE",
            ),
            make_candidate(
                "gpl-2.0-plus",
                "any later version",
                "gpl-2.0-plus_2.RULE",
            ),
        ],
    }

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["gpl-2.0", "gpl-2.0-plus"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert "{{GNU General Public License version 2}}" in rule.text
    assert "{{any later version}}" in rule.text


def test_add_required_phrases_backtracks_to_an_earlier_key_candidate():
    rule = Rule(
        license_expression="license-a AND license-b",
        identifier="candidate_backtracking_test.RULE",
        text="Alpha Long License and Backup Terms.",
        is_license_notice=True,
    )
    required_phrases_by_key = {
        "license-a": [
            make_candidate("license-a", "Alpha Long License", "license-a_1.RULE"),
            make_candidate("license-a", "Backup Terms", "license-a_2.RULE"),
        ],
        "license-b": [
            make_candidate("license-b", "Long License", "license-b_1.RULE"),
        ],
    }

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["license-a", "license-b"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert "{{Long License}}" in rule.text
    assert "{{Backup Terms}}" in rule.text
    assert "{{Alpha Long License}}" not in rule.text


def test_add_required_phrases_does_not_partially_mark_repeated_overlaps():
    text = "Alpha License terms and Alpha License."
    rule = Rule(
        license_expression="license-a AND license-b",
        identifier="repeated_overlap_test.RULE",
        text=text,
        is_license_notice=True,
    )
    required_phrases_by_key = {
        "license-a": [
            make_candidate("license-a", "Alpha License", "license-a_1.RULE"),
        ],
        "license-b": [
            make_candidate("license-b", "License terms", "license-b_1.RULE"),
        ],
    }

    composite_rules.add_required_phrases_to_composite_rules(
        rules=[rule],
        license_keys=["license-a", "license-b"],
        required_phrases_by_key=required_phrases_by_key,
        dry_run=True,
    )

    assert rule.text == text
    assert rule.source is None


def test_update_composite_rules_uses_single_key_required_phrases(monkeypatch):
    required_rules = {
        "mit": [make_required_phrase_rule("mit", "MIT License", "mit_1.RULE")],
        "apache-2.0": [
            make_required_phrase_rule(
                "apache-2.0",
                "Apache License",
                "apache-2.0_1.RULE",
            ),
        ],
    }
    target = Rule(
        license_expression="mit AND apache-2.0",
        identifier="mit_and_apache_test.RULE",
        text="Licensed under the MIT License and the Apache License.",
        is_license_notice=True,
    )

    licenses_by_key = {
        "mit": make_license(),
        "apache-2.0": make_license(),
    }

    monkeypatch.setattr(composite_rules, "get_licenses_db", lambda: licenses_by_key)
    monkeypatch.setattr(
        composite_rules,
        "get_base_rules_by_expression",
        lambda license_expression=None: required_rules,
    )
    monkeypatch.setattr(
        composite_rules,
        "get_updatable_rules_by_expression",
        lambda license_expression=None, simple_expression=True: {
            "mit AND apache-2.0": [target],
        },
    )

    composite_rules.update_composite_rules_using_required_phrases(dry_run=True)

    assert "{{MIT License}}" in target.text
    assert "{{Apache License}}" in target.text


def test_update_composite_rules_skips_generic_keys(monkeypatch):
    required_rules = {
        "mit": [make_required_phrase_rule("mit", "MIT License", "mit_1.RULE")],
        "unknown": [
            make_required_phrase_rule("unknown", "Unknown License", "unknown_1.RULE"),
        ],
    }
    target = Rule(
        license_expression="mit AND unknown",
        identifier="mit_and_unknown_test.RULE",
        text="Licensed under the MIT License and an Unknown License.",
        is_license_notice=True,
    )
    licenses_by_key = {
        "mit": make_license(),
        "unknown": make_license(is_generic=True),
    }

    monkeypatch.setattr(composite_rules, "get_licenses_db", lambda: licenses_by_key)
    monkeypatch.setattr(
        composite_rules,
        "get_base_rules_by_expression",
        lambda license_expression=None: required_rules,
    )
    monkeypatch.setattr(
        composite_rules,
        "get_updatable_rules_by_expression",
        lambda license_expression=None, simple_expression=True: {
            "mit AND unknown": [target],
        },
    )

    composite_rules.update_composite_rules_using_required_phrases(dry_run=True)

    assert "{{MIT License}}" in target.text
    assert "{{Unknown License}}" not in target.text


def test_composite_command_calls_update_and_validation(monkeypatch):
    update_calls = []
    validation_calls = []

    monkeypatch.setattr(
        composite_rules,
        "update_composite_rules_using_required_phrases",
        lambda **kwargs: update_calls.append(kwargs),
    )
    monkeypatch.setattr(
        composite_rules,
        "validate_and_reindex",
        lambda **kwargs: validation_calls.append(kwargs),
    )

    result = CliRunner().invoke(
        composite_rules.add_composite_required_phrases,
        [
            "--license-expression",
            "mit AND apache-2.0",
            "--write-phrase-source",
            "--validate",
            "--reindex",
            "--dry-run",
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    assert update_calls == [
        {
            "license_expression": "mit AND apache-2.0",
            "write_phrase_source": True,
            "dry_run": True,
            "verbose": True,
        },
    ]
    assert validation_calls == [
        {
            "validate": True,
            "reindex": True,
            "verbose": True,
        },
    ]
