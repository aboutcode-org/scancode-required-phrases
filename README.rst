ScanCode Required Phrases
=========================

``scancode-required-phrases`` provides commands for working with required
phrases in ScanCode license rules. ScanCode Toolkit provides the rule models,
tokenization, matching, and validation used by these commands.

Installation
============

Install this package in a Python 3.10 or newer environment:

.. code-block:: console

    python -m pip install .

Build a dataset
===============

.. code-block:: console

    build-required-phrases-dataset --rules-dir path/to/rules --output-dir dataset-output

The command writes ``train.jsonl``, ``val.jsonl``, and ``test.jsonl``. If
``--rules-dir`` is omitted, the installed ScanCode rules directory is used.

Update composite rules
======================

.. code-block:: console

    add-composite-required-phrases --dry-run --verbose

The command uses existing required phrases from single-key rules to update
composite rules. A rule is updated only when every relevant license key has a
non-overlapping match. It operates on the installed ScanCode rules directory.

Train and export a model
========================

Install the optional training dependencies and train from a generated dataset:

.. code-block:: console

    python -m pip install ".[training,training-8bit]"
    train-required-phrase-model --data-dir dataset-output --output-dir model-output \
        --model-revision 64a8c8eab3e352a784c658aef62be1662607476f \
        --optimizer adamw-8bit

Test-set evaluation is opt-in and should only be used for a final selected run.
The completed model is written to ``model-output/final-model`` only after local
reload and validation succeed. See ``docs/source/training.rst`` for details.

Run read-only prediction
========================

Install the inference dependencies, then load a validated final model and
return candidate required phrases without changing a ScanCode rule or file:

.. code-block:: console

    python -m pip install ".[inference]"

.. code-block:: python

    from scancode_required_phrases.inference import RequiredPhrasePredictor

    predictor = RequiredPhrasePredictor.from_model_dir("model-output/final-model")
    result = predictor.predict("Permission is hereby granted ...")

Predictions require human review before they are added to license rules.

Review model-predicted phrases
==============================

Install the inference dependencies and run the review command. It uses the
pinned public model by default:

.. code-block:: console

    python -m pip install ".[inference]"
    add-model-required-phrases --rule path/to/example.RULE

The default mode shows each phrase, model score, text context, and exact rule
diff before asking for approval. Decisions are saved in a resumable session.
Use ``--rules-dir`` for top-level rule files in a directory or ``--all`` for
eligible installed ScanCode rules.

Read-only prediction never creates a session or changes a rule:

.. code-block:: console

    add-model-required-phrases --rule path/to/example.RULE --predict-only

A wrong required phrase can cause a false negative. Batch processing therefore
requires explicit score thresholds and ``--yes`` before any write. Rules with
pending phrases are deferred unchanged while fully decided rules can be applied.
``--dry-run`` always writes zero rules. Run ``scancode-reindex-licenses`` after
changing installed ScanCode rules. Use ``--model`` for a custom local or remote
model.

Development
===========

Create a development environment and run the tests:

.. code-block:: console

    configure --dev
    venv\Scripts\pytest

On POSIX systems, run ``./configure --dev`` and ``venv/bin/pytest``.

See ``docs/source/dataset.rst`` and ``docs/source/composite_rules.rst`` for
details.
