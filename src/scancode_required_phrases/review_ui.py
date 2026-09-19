# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Display and collect model prediction reviews."""

import difflib
import os
from pathlib import Path
import re
import shlex
import shutil
from subprocess import list2cmdline
import textwrap

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


def marked_phrase(text):
    """Return phrase text displayed as a proposed required phrase."""
    return f"{{{{{text}}}}}"


def displayed_expression(expression, verbose):
    """Return an expression suitable for the current terminal."""
    if verbose:
        return expression
    width = max(4, shutil.get_terminal_size((100, 20)).columns - len("Expression: "))
    if len(expression) <= width:
        return expression
    return f"{expression[: width - 3]}..."


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
        before.decode("utf-8").splitlines(),
        after.decode("utf-8").splitlines(),
        fromfile=f"a/{rule_path.name}",
        tofile=f"b/{rule_path.name}",
        n=3,
        lineterm="",
    )
    for line in lines:
        if color and line.startswith("+") and not line.startswith("+++"):
            line = re.sub(
                r"{{.*?}}",
                lambda match: click.style(match.group(), fg="green"),
                line,
            )
        click.echo(line, color=color)


def phrase_context(rule, prediction):
    """Return wrapped ScanCode-tokenized context around a prediction."""
    words = words_from_text(rule.text)
    start = prediction["start_word"]
    end = prediction["end_word"]
    context = " ".join(words[max(0, start - 8) : end + 9])
    width = max(10, shutil.get_terminal_size((100, 20)).columns)
    return textwrap.fill(
        context,
        width=width,
        initial_indent="  ",
        subsequent_indent="  ",
    )


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
    phrase = marked_phrase(prediction["text"])
    phrase = click.style(phrase, fg="green", bold=True) if color else phrase

    click.echo(f"\nPrediction {position} of {total}\n", color=color)
    click.echo(f"Rule:       {record['identifier']}")
    if verbose:
        click.echo(f"Path:       {rule_path}")
    expression = displayed_expression(record["license_expression"], verbose)
    click.echo(f"Expression: {expression}")

    click.echo("\nPredicted required phrase:")
    click.echo(f"  {phrase}", color=color)

    click.echo(f"\nModel score: {prediction['score']:.1%}")
    click.echo("Validation:  Ready for review")

    if record["truncated"]:
        warning = "Rule input was truncated. This prediction requires human review."
        click.echo(f"\nWarning:\n  {click.style(warning, fg='yellow')}", color=color)

    click.echo("\nContext:")
    click.echo(phrase_context(rule, prediction))

    click.echo("\nProposed change:")
    try:
        content = preview_update(rule_path, rule, record, prediction)
    except ValueError as error:
        click.echo(f"  Cannot approve with current decisions: {error}")
        return None
    print_diff(rule_path, rule_path.read_bytes(), content, color)
    return content


def edit_prediction(rule_path, rule, record, prediction, color, verbose):
    """Preview and approve a valid replacement phrase."""
    original = marked_phrase(prediction["predicted_text"])
    original = click.style(original, fg="green", bold=True) if color else original

    click.echo("\nChange required phrase\n")
    click.echo("Editing changes the phrase boundaries.")
    click.echo("\nOriginal prediction:")
    click.echo(f"  {original}", color=color)
    click.echo("\nNearby rule context:")
    click.echo(phrase_context(rule, prediction))
    if verbose:
        click.echo("\nComplete rule text:")
        click.echo(rule.text)

    replacement = click.prompt(
        "\nReplacement phrase from the rule",
        default="",
        show_default=False,
    ).strip()
    if not replacement:
        click.echo("Change cancelled.")
        return False

    issue = candidate_issue(rule, replacement)
    if issue:
        click.echo(f"The replacement is not a valid candidate: {issue}")
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
        click.echo(f"The replacement cannot be approved: {error}")
        return False

    click.echo("\nProposed change:")
    print_diff(rule_path, rule_path.read_bytes(), content, color)
    changed = replacement != prediction["predicted_text"]
    question = "Approve this changed phrase?" if changed else "Approve this phrase?"
    if not click.confirm(f"\n{question}", default=False):
        click.echo("Change cancelled.")
        return False

    prediction["text"] = replacement
    prediction["decision"] = APPROVED
    prediction["decision_source"] = "human"
    return True


def print_actions(approval_available=True):
    """Print the available review actions."""
    click.echo("\nActions:")
    if approval_available:
        click.echo("  [y] approve   [n] reject   [e] change")
    else:
        click.echo("  [n] reject   [e] change")
    click.echo("  [s] later     [q] save/quit   [?] help")


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
        approval_available = preview is not None
        print_actions(approval_available)
        while True:
            answer = click.prompt("Action", default="", show_default=False).strip().lower()
            if answer == "y":
                if preview is None:
                    click.echo("This phrase cannot be approved with the current decisions.")
                    continue
                prediction["decision"] = APPROVED
                prediction["decision_source"] = "human"
                write_session(session_path, metadata, records)
                click.echo("Approved.")
                break
            if answer == "n":
                prediction["decision"] = REJECTED
                prediction["decision_source"] = "human"
                write_session(session_path, metadata, records)
                click.echo("Rejected.")
                break
            if answer == "e":
                if edit_prediction(rule_path, rule, record, prediction, color, verbose):
                    write_session(session_path, metadata, records)
                    if prediction["text"] == prediction["predicted_text"]:
                        click.echo("Approved.")
                    else:
                        click.echo("Changed and approved.")
                    break
                continue
            if answer == "s":
                click.echo("Pending review.")
                break
            if answer == "q":
                return False
            if answer == "?":
                print_actions(approval_available)
                continue
            if approval_available:
                click.echo("Enter y, n, e, s, q, or ?.")
            else:
                click.echo("Enter n, e, s, q, or ?.")

    return True


def session_summary(metadata, records):
    """Return user-facing counts derived from a session."""
    predictions = [prediction for record in records for prediction in record["predictions"]]
    return {
        "rules_scanned": metadata["rules_scanned"],
        "rules_eligible": metadata["rules_eligible"],
        "rules_selected": metadata["rules_selected"],
        "rules_with_predictions": len(records),
        "predictions": len(predictions),
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
            prediction["decision"] == APPROVED and prediction["decision_source"] == "score"
            for prediction in predictions
        ),
        "rejected": sum(prediction["decision"] == REJECTED for prediction in predictions),
        "pending": sum(prediction_needs_review(metadata, prediction) for prediction in predictions),
        "below_threshold": sum(
            prediction_is_below_threshold(metadata, prediction) for prediction in predictions
        ),
        "blocked": sum(prediction["validation_issue"] is not None for prediction in predictions),
        "truncated": metadata["truncated_rules"],
    }


def print_counts(items):
    """Print aligned count labels and values."""
    width = max(len(label) for label, _value in items) + 1
    for label, value in items:
        click.echo(f"{label + ':':<{width}}  {value}")


def print_review_overview(metadata, records, verbose):
    """Print counts explaining the interactive review queue."""
    counts = session_summary(metadata, records)
    prediction_counts = []
    if counts["pending"]:
        prediction_counts.append(f"{counts['pending']} ready for review")
    if counts["blocked"]:
        prediction_counts.append(f"{counts['blocked']} blocked by validation")

    items = [
        ("Selected rules", counts["rules_selected"]),
        ("Predictions", ", ".join(prediction_counts)),
    ]
    if verbose:
        items.extend(
            [
                ("Rules scanned", counts["rules_scanned"]),
                ("Rules eligible", counts["rules_eligible"]),
                ("Rules with predictions", counts["rules_with_predictions"]),
                ("Predictions found", counts["predictions"]),
                ("Truncated rules", counts["truncated"]),
            ]
        )
    click.echo("\nReview\n")
    print_counts(items)


def print_session_status(metadata, records, verbose):
    """Print the current decisions when review stops early."""
    counts = session_summary(metadata, records)
    items = [("Selected rules", counts["rules_selected"])]
    phrase_items = [
        ("Approved phrases", counts["approved"]),
        ("Changed and approved", counts["edited"]),
        ("Rejected", counts["rejected"]),
        ("Pending review", counts["pending"]),
        ("Blocked by validation", counts["blocked"]),
    ]
    if metadata["run_mode"] == "batch":
        phrase_items.extend(
            [
                ("Automatically approved", counts["auto_approved"]),
                ("Below review threshold", counts["below_threshold"]),
            ]
        )
    items.extend(
        phrase_items if verbose else [(label, value) for label, value in phrase_items if value]
    )
    if counts["truncated"] or verbose:
        items.append(("Truncated rules", counts["truncated"]))
    if verbose:
        items.extend(
            [
                ("Rules scanned", counts["rules_scanned"]),
                ("Rules eligible", counts["rules_eligible"]),
                ("Rules with predictions", counts["rules_with_predictions"]),
                ("Predictions found", counts["predictions"]),
            ]
        )
    click.echo("\nSession status\n")
    print_counts(items)


def print_summary(metadata, records, ready, deferred, unchanged, written, verbose):
    """Print one final summary for a preflighted session."""
    counts = session_summary(metadata, records)
    rule_items = [
        ("Selected rules", counts["rules_selected"]),
        ("Ready rules", ready),
        ("Deferred rules", deferred),
        ("Written rules", written),
    ]
    if verbose:
        rule_items = [
            ("Rules scanned", counts["rules_scanned"]),
            ("Rules eligible", counts["rules_eligible"]),
            ("Selected rules", counts["rules_selected"]),
            ("Rules with predictions", counts["rules_with_predictions"]),
            ("Ready rules", ready),
            ("Deferred rules", deferred),
            ("Unchanged rules", unchanged),
            ("Written rules", written),
        ]

    phrase_items = [
        ("Approved phrases", counts["approved"]),
        ("Changed and approved", counts["edited"]),
        ("Rejected", counts["rejected"]),
        ("Pending review", counts["pending"]),
        ("Blocked by validation", counts["blocked"]),
    ]
    if metadata["run_mode"] == "batch":
        phrase_items.extend(
            [
                ("Automatically approved", counts["auto_approved"]),
                ("Below review threshold", counts["below_threshold"]),
            ]
        )
    phrase_items.append(("Truncated rules", counts["truncated"]))
    if not verbose:
        phrase_items = [(label, value) for label, value in phrase_items if value]

    title = "Batch result" if metadata["run_mode"] == "batch" else "Summary"
    click.echo(f"\n{title}\n")
    print_counts(rule_items + phrase_items)

    if metadata["run_mode"] != "batch":
        return

    pending = [
        (record, prediction)
        for record in records
        for prediction in record["predictions"]
        if prediction_needs_review(metadata, prediction)
    ]
    review_score = metadata["review_score"]
    auto_score = metadata["auto_score"]
    score_pending = sum(
        review_score <= prediction["score"] < auto_score for _record, prediction in pending
    )
    truncated_pending = sum(
        prediction["score"] >= auto_score and record["truncated"] for record, prediction in pending
    )
    if score_pending == 1:
        click.echo(
            "\n1 prediction remains pending because its score is at least "
            "--review-score and below --auto-score."
        )
    elif score_pending:
        click.echo(
            f"\n{score_pending} predictions remain pending because their scores are at least "
            "--review-score and below --auto-score."
        )
    if truncated_pending:
        noun = "prediction" if truncated_pending == 1 else "predictions"
        verb = "remains" if truncated_pending == 1 else "remain"
        click.echo(
            f"\n{truncated_pending} high-scoring {noun} {verb} pending because truncated "
            "rule input cannot be automatically approved."
        )
    if deferred and (score_pending or truncated_pending):
        noun = "rule was" if deferred == 1 else "rules were"
        click.echo(
            f"{deferred} {noun} deferred because pending predictions defer the complete rule."
        )
