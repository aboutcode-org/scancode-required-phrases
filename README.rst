ScanCode Required Phrases
=========================

``scancode-required-phrases`` provides commands for working with required
phrases in ScanCode license rules. ScanCode Toolkit provides the rule models,
tokenization, matching, and validation used by these commands.

Installation
============

From this checkout, create the virtual environment and install the package
with Python 3.10 or newer:

.. code-block:: console

    ./configure
    venv/bin/build-required-phrases-dataset --help

On Windows, use ``configure.bat`` and ``venv\Scripts`` instead. The
``configure`` scripts install the package using the constraints in
``requirements.txt``.

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

Interactive review is the default. It shows each proposed phrase, its model
score and validation result, nearby context, and the exact rule diff before
prompting for a decision. Decisions are saved in a resumable session. Use
``--rules-dir`` for top-level rule files in a directory or ``--all`` for
eligible installed ScanCode rules. ``--limit`` limits selected rules, not
predictions.

Read-only prediction groups phrases by rule and never creates a session or
changes a rule:

.. code-block:: console

    add-model-required-phrases --rule path/to/example.RULE --predict-only

A wrong required phrase can cause a false negative. Batch processing does not
prompt and requires explicit score thresholds. ``--yes`` permits batch writes.
Rules with pending phrases are deferred unchanged while fully decided rules can
be applied. Interactive application uses one final confirmation. ``--dry-run``
prevents all rule-file writes while retaining the session. Resume uses its saved
predictions without loading the model. After the command writes installed
rules, it prints ``scancode-reindex-licenses`` as the next step. Use ``--model``
for a custom local or remote model.

Development
===========

Create a development environment and run the tests:

.. code-block:: console

    ./configure --dev
    venv/bin/pytest -q tests/test_dataset.py tests/test_composite_rules.py

On Windows, run ``configure.bat --dev`` and ``venv\Scripts\pytest``. This
installs the development extras with ``requirements-dev.txt`` constraints.

See ``docs/source/dataset.rst`` and ``docs/source/composite_rules.rst`` for
details.
