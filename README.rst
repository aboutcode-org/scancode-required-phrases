ScanCode Required Phrases
=========================

``scancode-required-phrases`` provides tools for working with required phrases
in ScanCode license rules.

The first available command builds a BIOES training dataset from required
phrases already marked with ``{{ }}`` in ScanCode ``.RULE`` files. ScanCode
Toolkit provides the rule models, tokenization, and validation used by the
command.

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

Development
===========

Create a development environment and run the tests:

.. code-block:: console

    configure --dev
    venv\Scripts\pytest

On POSIX systems, run ``./configure --dev`` and ``venv/bin/pytest``.

See ``docs/source/dataset.rst`` for dataset details.
