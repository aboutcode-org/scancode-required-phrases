# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display and collect model prediction reviews."""

import difflib
import os
from pathlib import Path
import shlex
import shutil
from subprocess import list2cmdline

import click

from scancode_required_phrases.inference import words_from_text
from scancode_required_phrases.model_rules import candidate_issue
from scancode_required_phrases.model_rules import prepare_predicted_phrases
from scancode_required_phrases.model_rules import serialize_rule
from scancode_required_phrases.review import APPROVED
from scancode_required_phrases.review import prediction_is_below_threshold
from scancode_required_phrases.review import prediction_needs_review
from scancode_required_phrases.review import REJECTED
from scancode_required_phrases.review import write_session


def resume_command(session_path):
    """Return a command that resumes a session on the current platform."""
    command = ["add-model-required-phrases", "--resume", str(session_path)]
    if os.name == "nt":
        return list2cmdline(command)
    return shlex.join(command)


def approved_phrases(record, current_prediction=None, replacement=None):
    """Return approved phrase text, optionally including the current candidate."""
    phrases = [
        prediction["text"]
        for prediction in record["predictions"]
        if prediction["decision"] == APPROVED
    ]
    if current_prediction is not None:
        phrases.append(replacement or current_prediction["text"])
    return phrases


def preview_update(rule_path, rule, record, prediction, replacement=None):
    """Return exact serialized bytes for a cumulative candidate update."""
    phrases = approved_phrases(record, prediction, replacement)
    updated_rule = prepare_predicted_phrases(rule, phrases)
    return serialize_rule(updated_rule, rule_path)


def print_diff(rule_path, before, after, color):
    """Print relevant unified-diff hunks from exact rule bytes."""
    lines = difflib.unified_diff(
        before.decode("utf-8").splitlines(keepends=True),
        after.decode("utf-8").splitlines(keepends=True),
        fromfile=f"a/{rule_path.name}",
        tofile=f"b/{rule_path.name}",
        n=3,
    )
    colors = {"+": "green", "-": "red", "@": "cyan"}
    for line in lines:
        line = line.rstrip("\n")
        click.echo(
            click.style(line, fg=colors.get(line[:1])) if color else line,
            color=color,
        )


def phrase_context(rule, prediction, color):
    """Return short ScanCode-tokenized context around a prediction."""
    words = words_from_text(rule.text)
    start = prediction["start_word"]
    end = prediction["end_word"]
    before = " ".join(words[max(0, start - 8) : start])
    phrase = " ".join(words[start : end + 1])
    after = " ".join(words[end + 1 : end + 9])
    phrase = click.style(phrase, bold=True, fg="yellow") if color else phrase
    return " ".join(part for part in (before, phrase, after) if part)


def show_prediction(
    rule_path,
    rule,
    record,
    prediction,
    position,
    total,
    color,
    verbose,
):
    """Show one pending prediction and its cumulative diff."""
    identifier = click.style(record["identifier"], bold=True) if color else record["identifier"]
    phrase = click.style(prediction["text"], fg="cyan", bold=True) if color else prediction["text"]
    score = f"{prediction['score']:.1%}"
    score = click.style(score, fg="yellow", bold=True) if color else score
    click.echo(f"\n[{position}/{total}] {identifier}", color=color)
    click.echo(f"{phrase}  {score}", color=color)

    expression = record["license_expression"]
    if verbose:
        click.echo(f"path: {rule_path}")
        click.echo(f"expression: {expression}")
    else:
        width = max(40, shutil.get_terminal_size((100, 20)).columns - 12)
        if len(expression) > width:
            expression = f"{expression[: width - 3]}..."
        click.echo(f"expression: {expression}")
    if record["truncated"]:
        warning = click.style("warning: rule input was truncated", fg="yellow")
        click.echo(warning, color=color)
    click.echo(f"context: {phrase_context(rule, prediction, color)}", color=color)

    try:
        content = preview_update(rule_path, rule, record, prediction)
    except ValueError as error:
        click.echo(f"cannot approve with current decisions: {error}")
        return None
    print_diff(rule_path, rule_path.read_bytes(), content, color)
    return content


def edit_prediction(rule_path, rule, record, prediction, color):
    """Prompt for and approve a valid replacement phrase."""
    click.echo("\nrule text")
    click.echo(rule.text)
    replacement = click.prompt(
        "phrase, empty to cancel",
        default="",
        show_default=False,
    ).strip()
    if not replacement:
        return False
    issue = candidate_issue(rule, replacement)
    if issue:
        click.echo(f"The phrase is not a valid candidate: {issue}")
        return False
    try:
        content = preview_update(
            rule_path,
            rule,
            record,
            prediction,
            replacement=replacement,
        )
    except ValueError as error:
        click.echo(f"The phrase cannot be approved: {error}")
        return False

    print_diff(rule_path, rule_path.read_bytes(), content, color)
    prediction["text"] = replacement
    prediction["decision"] = APPROVED
    prediction["decision_source"] = "human"
    return True


def review_predictions(session_path, metadata, records, loaded_rules, color, verbose=False):
    """Review each actionable prediction once and return False when the user quits."""
    pending = [
        (record, prediction)
        for record in records
        for prediction in record["predictions"]
        if prediction_needs_review(metadata, prediction)
    ]
    total = len(pending)

    for position, (record, prediction) in enumerate(pending, 1):
        rule_path = Path(record["path"])
        rule = loaded_rules[record["path"]]
        preview = show_prediction(
            rule_path,
            rule,
            record,
            prediction,
            position,
            total,
            color,
            verbose,
        )
        while True:
            answer = click.prompt(
                "[y] approve  [n] reject  [e] edit  [s] skip  [q] quit  [?] help",
                default="",
                show_default=False,
            ).strip().lower()
            if answer == "y":
                if preview is None:
                    click.echo("This phrase cannot be approved with the current decisions.")
                    continue
                prediction["decision"] = APPROVED
                prediction["decision_source"] = "human"
                write_session(session_path, metadata, records)
                click.echo("Approved")
                break
            if answer == "n":
                prediction["decision"] = REJECTED
                prediction["decision_source"] = "human"
                write_session(session_path, metadata, records)
                click.echo("Rejected")
                break
            if answer == "e":
                if edit_prediction(rule_path, rule, record, prediction, color):
                    write_session(session_path, metadata, records)
                    click.echo("Approved edited phrase")
                    break
                continue
            if answer == "s":
                click.echo("Skipped for this review pass")
                break
            if answer == "q":
                return False
            if answer == "?":
                click.echo("Approve, reject, edit, skip for later, or save and quit.")
                continue
            click.echo("Enter y, n, e, s, q, or ?.")

    return True


def session_summary(metadata, records):
    """Return user-facing counts derived from a session."""
    predictions = [
        prediction
        for record in records
        for prediction in record["predictions"]
    ]
    deferred = sum(
        record["applied_hash"] is None
        and any(
            prediction_needs_review(metadata, prediction)
            for prediction in record["predictions"]
        )
        for record in records
    )
    ready = sum(
        record["applied_hash"] is None
        and not any(
            prediction_needs_review(metadata, prediction)
            for prediction in record["predictions"]
        )
        and any(
            prediction["decision"] == APPROVED
            for prediction in record["predictions"]
        )
        for record in records
    )
    return {
        "rules_scanned": metadata["rules_scanned"],
        "rules_eligible": metadata["rules_eligible"],
        "rules_with_predictions": len(records),
        "rules_ready": ready,
        "rules_deferred": deferred,
        "approved": sum(
            prediction["decision"] == APPROVED
            and prediction["decision_source"] == "human"
            and prediction["text"] == prediction["predicted_text"]
            for prediction in predictions
        ),
        "edited": sum(
            prediction["decision"] == APPROVED
            and prediction["text"] != prediction["predicted_text"]
            for prediction in predictions
        ),
        "auto_approved": sum(
            prediction["decision"] == APPROVED
            and prediction["decision_source"] == "score"
            for prediction in predictions
        ),
        "rejected": sum(prediction["decision"] == REJECTED for prediction in predictions),
        "pending": sum(
            prediction_needs_review(metadata, prediction) for prediction in predictions
        ),
        "below_threshold": sum(
            prediction_is_below_threshold(metadata, prediction) for prediction in predictions
        ),
        "invalid": sum(prediction["validation_issue"] is not None for prediction in predictions),
        "truncated": metadata["truncated_rules"],
        "rules_written": sum(record["applied_hash"] is not None for record in records),
    }


def print_summary(metadata, records, unchanged=0):
    """Print concise session counts."""
    counts = session_summary(metadata, records)
    click.echo("\nSummary")
    click.echo(
        "Rules: "
        f"scanned {counts['rules_scanned']} | "
        f"ready {counts['rules_ready']} | "
        f"deferred {counts['rules_deferred']} | "
        f"unchanged {unchanged} | "
        f"written {counts['rules_written']}"
    )
    click.echo(
        "Phrases: "
        f"approved {counts['approved']} | "
        f"edited {counts['edited']} | "
        f"score-approved {counts['auto_approved']} | "
        f"rejected {counts['rejected']} | "
        f"pending {counts['pending']} | "
        f"below {counts['below_threshold']} | "
        f"invalid {counts['invalid']}"
    )
    if counts["truncated"]:
        click.echo(f"Truncated rules: {counts['truncated']}")
