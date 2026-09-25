# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Review and add model-predicted required phrases."""

import json
import os
from pathlib import Path
import sys
import warnings

import click

from licensedcode.models import InvalidRule
from licensedcode.models import rules_data_dir

from scancode_required_phrases.model_rules import load_prediction_rule
from scancode_required_phrases.model_rules import load_prediction_rules
from scancode_required_phrases.model_rules import load_predictor
from scancode_required_phrases.model_rules import predict_rule_candidates
from scancode_required_phrases.model_rules import select_installed_prediction_rules
from scancode_required_phrases.review import APPROVED
from scancode_required_phrases.review import create_metadata
from scancode_required_phrases.review import create_prediction
from scancode_required_phrases.review import create_rule_record
from scancode_required_phrases.review import create_session_path
from scancode_required_phrases.review import load_session_rules
from scancode_required_phrases.review import prepare_rule_updates
from scancode_required_phrases.review import prediction_needs_review
from scancode_required_phrases.review import read_session
from scancode_required_phrases.review import write_rule_updates
from scancode_required_phrases.review import write_session
from scancode_required_phrases.review_ui import displayed_expression
from scancode_required_phrases.review_ui import marked_phrase
from scancode_required_phrases.review_ui import print_counts
from scancode_required_phrases.review_ui import print_review_overview
from scancode_required_phrases.review_ui import print_session_status
from scancode_required_phrases.review_ui import print_summary
from scancode_required_phrases.review_ui import resume_command
from scancode_required_phrases.review_ui import review_predictions


DEFAULT_MODEL = "Kaushik-Kumar-CEG/scancode-required-phrases-deberta-bioes-crf-hardened"
DEFAULT_MODEL_REVISION = "11215925b0f9b64cfcfbbb5492b52d6aeb5a572b"


def stdin_is_tty():
    """Return whether interactive input is available."""
    return sys.stdin.isatty()


def stdout_is_tty():
    return click.get_text_stream("stdout").isatty()


def validate_options(
    rule_path,
    rules_directory,
    all_rules,
    license_expression,
    limit,
    predict_only,
    batch,
    resume,
    model,
    model_revision,
    auto_score,
    review_score,
    yes,
    session_path,
    json_output,
):
    """Validate command options before loading rules or a model."""
    if resume:
        conflicts = (
            rule_path,
            rules_directory,
            all_rules,
            license_expression,
            limit,
            predict_only,
            batch,
            model,
            model_revision,
            auto_score,
            review_score,
            yes,
            session_path,
            json_output,
        )
        if any(value not in (None, False, 0) for value in conflicts):
            raise click.UsageError("--resume cannot be used with new-run options")
        if not stdin_is_tty():
            raise click.UsageError("--resume requires an interactive terminal")
        return

    targets = (rule_path is not None, rules_directory is not None, all_rules)
    if sum(targets) != 1:
        raise click.UsageError("Use exactly one of --rule, --rules-dir, or --all")
    if predict_only and batch:
        raise click.UsageError("--predict-only and --batch are mutually exclusive")
    if json_output and not predict_only:
        raise click.UsageError("--json requires --predict-only")
    if session_path and predict_only:
        raise click.UsageError("--session cannot be used with --predict-only")
    if session_path and (Path(session_path).exists() or Path(session_path).is_symlink()):
        raise click.UsageError(f"Session file already exists: {session_path}")

    if batch:
        if auto_score is None or review_score is None:
            raise click.UsageError("--batch requires --auto-score and --review-score")
        if review_score > auto_score:
            raise click.UsageError("--review-score cannot exceed --auto-score")
    elif auto_score is not None or review_score is not None or yes:
        raise click.UsageError("Score thresholds and --yes require --batch")

    if not predict_only and not batch and not stdin_is_tty():
        raise click.UsageError("Interactive review requires an interactive terminal")


def resolve_model(model, model_revision):
    """Return the requested model or the pinned project default."""
    if not model:
        return DEFAULT_MODEL, model_revision or DEFAULT_MODEL_REVISION
    if model == DEFAULT_MODEL and not model_revision:
        return model, DEFAULT_MODEL_REVISION
    return model, model_revision


def select_targets(rule_path, rules_directory, all_rules, license_expression, limit):
    """Return target metadata, selected rules, and selection counts."""
    if rule_path:
        eligible = [load_prediction_rule(rule_path, license_expression)]
        target_mode = "rule"
        target = eligible[0][0]
        rules_scanned = 1
    elif rules_directory:
        root = Path(rules_directory).resolve(strict=True)
        rules_scanned = len(list(root.glob("*.RULE")))
        eligible = load_prediction_rules(root, license_expression)
        target_mode = "rules_dir"
        target = root
    else:
        target = Path(rules_data_dir).resolve(strict=True)
        rules_scanned = len(list(target.glob("*.RULE")))
        eligible = select_installed_prediction_rules(license_expression)
        target_mode = "all"

    rules_eligible = len(eligible)
    selected = eligible[:limit] if limit else eligible
    return target_mode, target, selected, rules_scanned, rules_eligible


def predict_targets(
    selected,
    predictor,
    batch=False,
    auto_score=None,
    progress_file=None,
):
    """Return session records, output rows, and the truncated rule count."""
    records = []
    rows = []
    truncated_rules = 0
    with click.progressbar(
        selected,
        label="Predicting rules",
        file=progress_file,
    ) as progress:
        for rule_path, rule in progress:
            result, candidates = predict_rule_candidates(rule, predictor)
            if result.truncated:
                truncated_rules += 1
            predictions = []
            for prediction, issue in candidates:
                entry = create_prediction(prediction, issue)
                if (
                    batch
                    and issue is None
                    and not result.truncated
                    and prediction.score >= auto_score
                ):
                    entry["decision"] = APPROVED
                    entry["decision_source"] = "score"
                predictions.append(entry)
                rows.append(
                    {
                        "path": str(rule_path),
                        "identifier": rule.identifier,
                        "license_expression": rule.license_expression,
                        "phrase": prediction.text,
                        "score": prediction.score,
                        "start_word": prediction.start_word,
                        "end_word": prediction.end_word,
                        "truncated": result.truncated,
                        "validation_issue": issue,
                    }
                )
            if predictions:
                records.append(
                    create_rule_record(
                        rule_path=rule_path,
                        rule=rule,
                        predictions=predictions,
                        truncated=result.truncated,
                    )
                )
    return records, rows, truncated_rules


def write_json_predictions(rows, json_output):
    """Write prediction rows as JSON to a file or stdout."""
    content = json.dumps(rows, ensure_ascii=False, allow_nan=False, indent=2)
    if json_output == "-":
        click.echo(content)
        return
    output_path = Path(json_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content + "\n", encoding="utf-8")


VALIDATION_MESSAGES = {
    None: "Ready for review",
    "rejected": "Blocked - does not meet ScanCode required-phrase checks",
    "not_found": "Blocked - exact phrase was not found in the rule text",
    "ambiguous": "Blocked - phrase appears more than once",
}


def print_prediction_rows(
    rows,
    rules_scanned,
    rules_eligible,
    rules_selected,
    truncated_rules,
    verbose,
    color,
):
    """Print read-only predictions grouped by rule."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["path"], []).append(row)

    ready = sum(row["validation_issue"] is None for row in rows)
    blocked = len(rows) - ready
    prediction_counts = []
    if ready:
        prediction_counts.append(f"{ready} ready")
    if blocked:
        prediction_counts.append(f"{blocked} blocked by validation")
    if not prediction_counts:
        prediction_counts.append("0")

    items = [
        ("Selected rules", rules_selected),
        ("Rules with predictions", len(grouped)),
        ("Predictions", ", ".join(prediction_counts)),
    ]
    if verbose:
        items.extend(
            [
                ("Rules scanned", rules_scanned),
                ("Rules eligible", rules_eligible),
                ("Predictions found", len(rows)),
                ("Truncated rules", truncated_rules),
            ]
        )
    click.echo("\nPrediction overview\n")
    print_counts(items)
    if not rows:
        click.echo("\nNo model predictions found.")
        return

    for position, predictions in enumerate(grouped.values(), 1):
        first = predictions[0]
        click.echo(f"\nRule {position} of {len(grouped)}\n")
        click.echo(f"Rule:       {first['identifier']}")
        if verbose:
            click.echo(f"Path:       {first['path']}")
        expression = displayed_expression(first["license_expression"], verbose)
        click.echo(f"Expression: {expression}")
        click.echo("\nPredicted required phrases:")
        for prediction_number, row in enumerate(predictions, 1):
            phrase = marked_phrase(row["phrase"])
            phrase = click.style(phrase, fg="green", bold=True) if color else phrase
            click.echo(f"  {prediction_number}. {phrase}", color=color)
            click.echo(f"     Model score: {row['score']:.1%}")
            click.echo(f"     Validation: {VALIDATION_MESSAGES[row['validation_issue']]}")
            if row["truncated"]:
                click.echo("     Warning: Rule input was truncated.")


def finish_session(session_path, metadata, records, dry_run, allow_write, verbose):
    """Preflight a complete session and optionally write its rules."""
    loaded_rules, reconciled = load_session_rules(metadata, records)
    if reconciled:
        click.echo(f"Recovered interrupted writes: {', '.join(reconciled)}")
        write_session(session_path, metadata, records)

    work, unchanged, deferred = prepare_rule_updates(metadata, records, loaded_rules)
    write_session(session_path, metadata, records)
    if work:
        click.echo(f"\nReady rule paths ({len(work)}):")
        for _record, rule_path, _content in work:
            click.echo(f"  {rule_path}")

    written = []
    should_write = False
    if not dry_run:
        if allow_write is None and work:
            noun = "rule" if len(work) == 1 else "rules"
            should_write = click.confirm(
                f"\nApply updates to {len(work)} {noun}?",
                default=False,
            )
        elif allow_write:
            should_write = bool(work)

    if should_write:
        written = write_rule_updates(session_path, metadata, records, work)

    print_summary(
        metadata,
        records,
        ready=len(work),
        deferred=len(deferred),
        unchanged=len(unchanged),
        written=len(written),
        verbose=verbose,
    )
    if dry_run:
        click.echo("\nDry run: no rule files were written.")
    elif metadata["run_mode"] == "batch" and work and allow_write is False:
        click.echo("\nReady rules were not written because --yes was not provided.")
    click.echo(f"\nSession: {session_path}")

    has_unapplied_work = bool(work and not written)
    if deferred or has_unapplied_work:
        click.echo(f"Resume: {resume_command(session_path)}")
    if metadata["target_mode"] == "all" and written:
        click.echo("\nNext step:")
        click.echo("  scancode-reindex-licenses")


def run_new(
    rule_path,
    rules_directory,
    all_rules,
    license_expression,
    limit,
    predict_only,
    batch,
    model,
    model_revision,
    auto_score,
    review_score,
    yes,
    session_path,
    json_output,
    dry_run,
    verbose,
    color,
):
    """Run prediction and the selected new-run mode."""
    progress_file = (
        click.get_text_stream("stderr") if json_output == "-" else click.get_text_stream("stdout")
    )
    click.echo("Selecting rules...", file=progress_file)
    target_mode, target, selected, rules_scanned, rules_eligible = select_targets(
        rule_path,
        rules_directory,
        all_rules,
        license_expression,
        limit,
    )
    rules_selected = len(selected)
    if not selected:
        if predict_only and json_output:
            write_json_predictions([], json_output)
        else:
            print_prediction_rows(
                [],
                rules_scanned=rules_scanned,
                rules_eligible=rules_eligible,
                rules_selected=rules_selected,
                truncated_rules=0,
                verbose=verbose,
                color=color,
            )
        click.echo("No eligible rules found.", err=json_output == "-")
        return

    noun = "rule" if rules_selected == 1 else "rules"
    click.echo(f"Selected {rules_selected} {noun}.", file=progress_file)
    click.echo("Checking model files...", file=progress_file)

    def report_model_load():
        click.echo("Loading model into memory...", file=progress_file)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.script` is deprecated\..*",
            category=FutureWarning,
            module=r"torch\.jit\._script",
        )
        predictor = load_predictor(
            model,
            hf_token=os.environ.get("HF_TOKEN"),
            revision=model_revision,
            before_model_load=report_model_load,
        )
    click.echo("Model ready.", file=progress_file)
    records, rows, truncated_rules = predict_targets(
        selected,
        predictor,
        batch=batch,
        auto_score=auto_score,
        progress_file=progress_file,
    )

    if predict_only:
        if json_output:
            write_json_predictions(rows, json_output)
        else:
            print_prediction_rows(
                rows,
                rules_scanned=rules_scanned,
                rules_eligible=rules_eligible,
                rules_selected=rules_selected,
                truncated_rules=truncated_rules,
                verbose=verbose,
                color=color,
            )
        return
    if not rows:
        print_prediction_rows(
            [],
            rules_scanned=rules_scanned,
            rules_eligible=rules_eligible,
            rules_selected=rules_selected,
            truncated_rules=truncated_rules,
            verbose=verbose,
            color=color,
        )
        return

    session_path = create_session_path(session_path)
    metadata = create_metadata(
        model=model,
        model_revision=model_revision,
        target_mode=target_mode,
        target=target,
        run_mode="batch" if batch else "interactive",
        rules_scanned=rules_scanned,
        rules_eligible=rules_eligible,
        rules_selected=rules_selected,
        truncated_rules=truncated_rules,
        auto_score=auto_score,
        review_score=review_score,
    )
    write_session(session_path, metadata, records)
    if verbose:
        click.echo(f"Session: {session_path}")

    if batch:
        finish_session(
            session_path,
            metadata,
            records,
            dry_run=dry_run,
            allow_write=yes,
            verbose=verbose,
        )
        return

    loaded_rules, reconciled = load_session_rules(metadata, records)
    if reconciled:
        write_session(session_path, metadata, records)
    print_review_overview(metadata, records, verbose)
    complete = review_predictions(
        session_path,
        metadata,
        records,
        loaded_rules,
        color,
        verbose,
    )
    if not complete:
        print_session_status(metadata, records, verbose)
        click.echo(f"\nSession: {session_path}")
        click.echo(f"Resume: {resume_command(session_path)}")
        return
    finish_session(
        session_path,
        metadata,
        records,
        dry_run=dry_run,
        allow_write=None,
        verbose=verbose,
    )


def run_resume(session_path, dry_run, color, verbose):
    """Resume review or application without running inference."""
    metadata, records = read_session(session_path)
    loaded_rules, reconciled = load_session_rules(metadata, records)
    if reconciled:
        click.echo(f"Recovered interrupted writes: {', '.join(reconciled)}")
        write_session(session_path, metadata, records)

    if any(
        prediction_needs_review(metadata, prediction)
        for record in records
        for prediction in record["predictions"]
    ):
        print_review_overview(metadata, records, verbose)
    complete = review_predictions(
        session_path,
        metadata,
        records,
        loaded_rules,
        color,
        verbose,
    )
    if not complete:
        print_session_status(metadata, records, verbose)
        click.echo(f"\nSession: {session_path}")
        click.echo(f"Resume: {resume_command(session_path)}")
        return
    finish_session(
        session_path,
        metadata,
        records,
        dry_run=dry_run,
        allow_write=None,
        verbose=verbose,
    )


@click.command(name="add-model-required-phrases")
@click.option(
    "--rule",
    "rule_path",
    type=click.Path(path_type=Path),
    help="Use one .RULE file.",
)
@click.option(
    "--rules-dir",
    "rules_directory",
    type=click.Path(path_type=Path),
    help="Use top-level .RULE files in a directory.",
)
@click.option("--all", "all_rules", is_flag=True, help="Use eligible installed rules.")
@click.option("-l", "--license-expression", help="Only use rules for this expression.")
@click.option(
    "--limit",
    default=0,
    type=click.IntRange(min=0),
    help="Limit selected rules; zero selects all.",
)
@click.option(
    "--predict-only", is_flag=True, help="Print predictions without prompting or writing rules."
)
@click.option("--batch", is_flag=True, help="Classify predictions without prompting.")
@click.option(
    "--resume",
    type=click.Path(path_type=Path),
    help="Resume saved predictions without loading the model.",
)
@click.option("--model", help="Local model or alternate Hugging Face repository.")
@click.option("--model-revision", help="Full commit hash for a remote model.")
@click.option(
    "--auto-score",
    type=click.FloatRange(min=0, max=1),
    help="Score at or above which valid batch predictions may be approved.",
)
@click.option(
    "--review-score",
    type=click.FloatRange(min=0, max=1),
    help="Score at or above which lower-scoring batch predictions remain pending.",
)
@click.option("--yes", is_flag=True, help="Permit batch writes after preflight.")
@click.option(
    "--session",
    "session_path",
    type=click.Path(path_type=Path),
    help="Use this path for a new review session.",
)
@click.option(
    "--json",
    "json_output",
    type=click.Path(path_type=str, allow_dash=True),
    help="Write predict-only JSON to a file or '-' for stdout.",
)
@click.option("--dry-run", is_flag=True, help="Prevent every rule-file write.")
@click.option("-v", "--verbose", is_flag=True, help="Print additional processing details.")
@click.help_option("-h", "--help")
def add_model_required_phrases(
    rule_path,
    rules_directory,
    all_rules,
    license_expression,
    limit,
    predict_only,
    batch,
    resume,
    model,
    model_revision,
    auto_score,
    review_score,
    yes,
    session_path,
    json_output,
    dry_run,
    verbose,
):
    """Review model-predicted required phrases and optionally add them to rules."""
    color = "NO_COLOR" not in os.environ and stdout_is_tty()
    try:
        validate_options(
            rule_path,
            rules_directory,
            all_rules,
            license_expression,
            limit,
            predict_only,
            batch,
            resume,
            model,
            model_revision,
            auto_score,
            review_score,
            yes,
            session_path,
            json_output,
        )
        if resume:
            run_resume(resume, dry_run=dry_run, color=color, verbose=verbose)
        else:
            model, model_revision = resolve_model(model, model_revision)
            run_new(
                rule_path,
                rules_directory,
                all_rules,
                license_expression,
                limit,
                predict_only,
                batch,
                model,
                model_revision,
                auto_score,
                review_score,
                yes,
                session_path,
                json_output,
                dry_run,
                verbose,
                color,
            )
    except (InvalidRule, OSError, ValueError) as error:
        raise click.ClickException(str(error)) from error
