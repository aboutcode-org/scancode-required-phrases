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
carefully and use disposable rule copies before changing installed data.

Interactive review
------------------

Review one rule:

.. code-block:: console

    add-model-required-phrases --rule path/to/example.RULE

Use ``--rules-dir`` to review sorted top-level ``.RULE`` files in a directory.
Use ``--all`` to review eligible rules installed with ScanCode Toolkit. The
command shows the rule, expression, predicted phrase, model score, context, and
exact diff. Approve, reject, edit, skip, or save and quit at each prompt.

Sessions are created automatically and retained for audit. The command prints an
exact resume command when review remains unfinished:

.. code-block:: console

    add-model-required-phrases --resume path/to/session.jsonl

Review and application are resumable. An interrupted prediction run starts
prediction again.

Read-only prediction
--------------------

Print validated and rejected model output without creating decisions or writing
rules:

.. code-block:: console

    add-model-required-phrases --rule path/to/example.RULE --predict-only

Use ``--json predictions.json`` for machine-readable output or ``--json -`` for
JSON on standard output.

Batch classification
--------------------

Batch mode has no default thresholds. Both values must be chosen explicitly:

.. code-block:: console

    add-model-required-phrases --all --batch \
        --auto-score 0.90 --review-score 0.70 --dry-run

Scores at or above ``--auto-score`` are staged for automatic approval only after
ScanCode validation. Scores from ``--review-score`` up to ``--auto-score`` stay
pending. Lower scores are ignored. Truncated rules always require review. A rule
with any pending phrase is deferred unchanged. Other fully decided rules may be
applied after complete preflight.

``--yes`` permits non-interactive writes for ready rules. ``--dry-run`` always
takes precedence and writes zero rules. Use ``--model`` and ``--model-revision``
to override the pinned public model.

Safe application
----------------

Review never mutates rule files. Before application, the command verifies every
path and file hash, prepares all approved updates, and displays a final summary.
Each changed rule is then written once with an atomic replacement at its exact
source path. Rejected and below-threshold phrases are not written.

After changing installed ScanCode rules, run:

.. code-block:: console

    scancode-reindex-licenses
