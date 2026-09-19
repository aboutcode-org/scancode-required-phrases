Review model-predicted required phrases
=======================================

Install the inference dependencies before using the command:

.. code-block:: console

    python -m pip install ".[inference]"

The public model is available without a Hugging Face token:

.. code-block:: text

    Kaushik-Kumar-CEG/scancode-required-phrases-deberta-bioes-crf-hardened
    11215925b0f9b64cfcfbbb5492b52d6aeb5a572b

A wrong required phrase can prevent a true license match. Review predictions
carefully and use disposable rule copies before changing installed data. Model
files downloaded from Hugging Face are cached on disk. Every new prediction
command still loads a new PyTorch model into that process's memory.

Interactive review
------------------

Review one rule:

.. code-block:: console

    add-model-required-phrases --rule path/to/example.RULE

Use ``--rules-dir`` to review sorted top-level ``.RULE`` files in a directory.
Use ``--all`` to review eligible rules installed with ScanCode Toolkit.
``--limit`` limits rules sent to the model, not the number of predictions.

Each review shows the proposed required phrase, model score, validation result,
nearby context, and exact unified diff. Approve, reject, change the phrase,
review it later, or save and quit. A changed phrase is validated and shown as a
new diff before approval. Use ``--verbose`` for additional selection details,
exact rule paths, and complete rule text while changing a phrase.

Sessions are created automatically and retained for audit. The command prints an
exact resume command when review remains unfinished:

.. code-block:: console

    add-model-required-phrases --resume path/to/session.jsonl

Review and application are resumable. ``--resume`` uses the stored predictions
without loading the model. An interrupted prediction run starts prediction
again.

Read-only prediction
--------------------

Print validated and rejected model output grouped by rule without creating
decisions or writing rules:

.. code-block:: console

    add-model-required-phrases --rule path/to/example.RULE --predict-only

Use ``--json predictions.json`` for machine-readable output or ``--json -`` for
JSON on standard output. In JSON, ``validation_issue`` is ``null`` when a
prediction passes validation. Predict-only is optional before interactive
review: interactive mode already predicts and reviews in one process. Running
predict-only and then starting interactive review in a new process loads the
model twice.

Batch classification
--------------------

Batch mode has no default thresholds. Both values must be chosen explicitly:

.. code-block:: console

    add-model-required-phrases --all --batch \
        --auto-score 0.90 --review-score 0.70 --dry-run

Valid predictions with scores at or above ``--auto-score`` may be approved
automatically. Scores at or above ``--review-score`` and below
``--auto-score`` stay pending. Lower scores are ignored rather than rejected.
Validation always takes precedence over score, and truncated rule input is
never approved automatically. A rule with any pending phrase is deferred
unchanged. Other fully decided rules may be applied after complete preflight.

``--yes`` permits non-interactive writes for ready rules. Without ``--yes``,
ready rules remain unwritten and the session can be resumed. ``--dry-run``
always takes precedence, prevents every rule-file write, and retains the
session. Use ``--model`` and ``--model-revision`` to override the pinned public
model.

Safe application
----------------

Review never mutates rule files. Before application, the command verifies every
path and file hash, prepares all approved updates, and shows their exact paths.
Interactive use asks once before applying them. Declining leaves the session
saved. Each changed rule is written once with an atomic replacement at its exact
source path. Rejected and below-threshold phrases are not written, and
``--dry-run`` reports that no rule files were written.

After the command writes installed ScanCode rules, it reports this next step:

.. code-block:: console

    scancode-reindex-licenses
