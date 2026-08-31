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
        --optimizer adamw-8bit

Test-set evaluation is opt-in and should only be used for a final selected run.
See ``docs/source/training.rst`` for training and ONNX export details.

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
