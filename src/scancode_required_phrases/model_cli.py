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
from scancode_required_phrases.review import read_session
from scancode_required_phrases.review import write_rule_updates
from scancode_required_phrases.review import write_session
from scancode_required_phrases.review_ui import print_summary
from scancode_required_phrases.review_ui import resume_command
from scancode_required_phrases.review_ui import review_predictions


DEFAULT_MODEL = "Kaushik-Kumar-CEG/scancode-required-phrases-deberta-bioes-crf-hardened"
DEFAULT_MODEL_REVISION = "11215925b0f9b64cfcfbbb5492b52d6aeb5a572b"


def stdin_is_tty():
    """Return whether interactive input is available."""
    return sys.stdin.isatty()


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
    """Return target metadata and selected rules for a new run."""
    if rule_path:
        selected = [load_prediction_rule(rule_path, license_expression)]
        target_mode = "rule"
        target = selected[0][0]
        rules_scanned = 1
    elif rules_directory:
        root = Path(rules_directory).resolve(strict=True)
        rules_scanned = len(list(root.glob("*.RULE")))
        selected = load_prediction_rules(root, license_expression)
        target_mode = "rules_dir"
        target = root
    else:
        selected = select_installed_prediction_rules(license_expression)
        target_mode = "all"
        target = Path(rules_data_dir).resolve(strict=True)
        rules_scanned = len(selected)

    if limit:
        selected = selected[:limit]
    return target_mode, target, selected, rules_scanned


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


def print_prediction_rows(rows):
    """Print concise read-only prediction output."""
    if not rows:
        click.echo("No model predictions found")
        return
    for row in rows:
        status = row["validation_issue"] or "valid"
        truncated = " truncated" if row["truncated"] else ""
        click.echo(
            f"{row['identifier']}  {row['license_expression']}  "
            f"{row['score']:.1%}  {status}{truncated}"
        )
        click.echo(f"  {row['phrase']}")


def finish_session(session_path, metadata, records, dry_run, allow_write):
    """Preflight a complete session and optionally write its rules."""
    loaded_rules, reconciled = load_session_rules(metadata, records)
    if reconciled:
        click.echo(f"Recovered interrupted writes: {', '.join(reconciled)}")
        write_session(session_path, metadata, records)

    work, unchanged, deferred = prepare_rule_updates(metadata, records, loaded_rules)
    write_session(session_path, metadata, records)
    print_summary(metadata, records, unchanged=len(unchanged))
    if not work:
        click.echo(f"Session saved: {session_path}")
        if deferred:
            click.echo(f"Resume with: {resume_command(session_path)}")
        return
    if dry_run:
        click.echo("Dry run: no rules were written")
        click.echo(f"Session saved: {session_path}")
        if deferred:
            click.echo(f"Resume with: {resume_command(session_path)}")
        return

    if allow_write is None:
        choice = click.prompt(
            "[d] dry-run  [a] apply  [s] save and exit",
            default="s",
            show_default=False,
            type=click.Choice(["d", "a", "s"], case_sensitive=False),
        ).lower()
        if choice != "a":
            if choice == "d":
                click.echo("Dry run: no rules were written")
            click.echo(f"Session saved: {session_path}")
            if deferred:
                click.echo(f"Resume with: {resume_command(session_path)}")
            return
    elif not allow_write:
        click.echo(f"Session saved: {session_path}")
        if deferred:
            click.echo(f"Resume with: {resume_command(session_path)}")
        return

    write_rule_updates(session_path, metadata, records, work)
    print_summary(metadata, records, unchanged=len(unchanged))
    if deferred:
        click.echo(f"Resume with: {resume_command(session_path)}")
    if metadata["target_mode"] == "all":
        click.echo("Run scancode-reindex-licenses to use the new required phrases")


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
        click.get_text_stream("stderr")
        if json_output == "-"
        else click.get_text_stream("stdout")
    )
    click.echo("Selecting rules...", file=progress_file)
    target_mode, target, selected, rules_scanned = select_targets(
        rule_path,
        rules_directory,
        all_rules,
        license_expression,
        limit,
    )
    if not selected:
        if predict_only and json_output:
            write_json_predictions([], json_output)
        click.echo("No eligible rules found", err=json_output == "-")
        return

    click.echo(f"Selected {len(selected)} eligible rules.", file=progress_file)
    click.echo("Loading model...", file=progress_file)
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
        )
    click.echo("Model loaded.", file=progress_file)
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
            print_prediction_rows(rows)
        return
    if not rows:
        click.echo("No model predictions found")
        return

    session_path = create_session_path(session_path)
    metadata = create_metadata(
        model=model,
        model_revision=model_revision,
        target_mode=target_mode,
        target=target,
        run_mode="batch" if batch else "interactive",
        rules_scanned=rules_scanned,
        rules_eligible=len(selected),
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
        )
        return

    loaded_rules, reconciled = load_session_rules(metadata, records)
    if reconciled:
        write_session(session_path, metadata, records)
    complete = review_predictions(
        session_path,
        metadata,
        records,
        loaded_rules,
        color,
        verbose,
    )
    if not complete:
        print_summary(metadata, records)
        click.echo(f"Resume with: {resume_command(session_path)}")
        return
    finish_session(
        session_path,
        metadata,
        records,
        dry_run=dry_run,
        allow_write=None,
    )


def run_resume(session_path, dry_run, color, verbose):
    """Resume review or application without running inference."""
    metadata, records = read_session(session_path)
    loaded_rules, reconciled = load_session_rules(metadata, records)
    if reconciled:
        click.echo(f"Recovered interrupted writes: {', '.join(reconciled)}")
        write_session(session_path, metadata, records)

    complete = review_predictions(
        session_path,
        metadata,
        records,
        loaded_rules,
        color,
        verbose,
    )
    if not complete:
        print_summary(metadata, records)
        click.echo(f"Resume with: {resume_command(session_path)}")
        return
    finish_session(
        session_path,
        metadata,
        records,
        dry_run=dry_run,
        allow_write=None,
    )


@click.command(name="add-model-required-phrases")
@click.option(
    "--rule",
    "rule_path",
    type=click.Path(path_type=Path),
    help="Review one .RULE file.",
)
@click.option(
    "--rules-dir",
    "rules_directory",
    type=click.Path(path_type=Path),
    help="Review top-level .RULE files in a directory.",
)
@click.option("--all", "all_rules", is_flag=True, help="Review eligible installed rules.")
@click.option("-l", "--license-expression", help="Only use rules for this expression.")
@click.option(
    "--limit",
    default=0,
    type=click.IntRange(min=0),
    help="Stop after this many eligible rules; zero uses all.",
)
@click.option("--predict-only", is_flag=True, help="Print predictions without decisions or writes.")
@click.option("--batch", is_flag=True, help="Classify predictions using explicit scores.")
@click.option(
    "--resume",
    type=click.Path(path_type=Path),
    help="Resume an existing review session.",
)
@click.option("--model", help="Local model or alternate Hugging Face repository.")
@click.option("--model-revision", help="Full commit hash for a remote model.")
@click.option(
    "--auto-score",
    type=click.FloatRange(min=0, max=1),
    help="Batch automatic-approval score.",
)
@click.option(
    "--review-score",
    type=click.FloatRange(min=0, max=1),
    help="Batch pending-review score.",
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
@click.option("--dry-run", is_flag=True, help="Validate and preview without writing rules.")
@click.option("--no-color", is_flag=True, help="Disable colored output.")
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
    no_color,
    verbose,
):
    """Review and add model-predicted required phrases to license rules."""
    color = (
        not no_color
        and "NO_COLOR" not in os.environ
        and click.get_text_stream("stdout").isatty()
    )
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


if __name__ == "__main__":
    add_model_required_phrases()
