Build a required phrase dataset
===============================

The ``build-required-phrases-dataset`` command reads ScanCode ``.RULE`` files
and creates a BIOES-labelled JSONL dataset from required phrases marked with
``{{ }}``.

Run the command
---------------

.. code-block:: console

    build-required-phrases-dataset \
        --rules-dir path/to/rules \
        --output-dir dataset-output

``--rules-dir`` defaults to the installed ScanCode rules directory.
``--output-dir`` defaults to ``dataset-output``.

Output
------

The output directory contains:

* ``train.jsonl``
* ``val.jsonl``
* ``test.jsonl``

Each record contains the rule identifier, license expression, rule type,
unmarked text, tokens, and BIOES labels.

Splitting
---------

Rules with common license expressions are assigned by a deterministic hash of
their identifiers. Rules with rarer expressions stay together in one split.
The target proportions are 80 percent training, 10 percent validation, and
10 percent test.

Only eligible rules containing a marked required phrase are included.
Dedicated ``is_required_phrase`` rules, deprecated rules, false positives,
license clues, license introductions, and rules without a license expression
are excluded.
