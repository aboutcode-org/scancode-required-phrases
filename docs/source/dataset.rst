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
unmarked text, tokens, and BIOES labels. For example, the marked rule text
``Use {{MIT}} or {{Apache License Version}}`` produces tokens ``Use``,
``MIT``, ``or``, ``Apache``, ``License``, ``Version`` with labels ``O``,
``S-REQ``, ``O``, ``B-REQ``, ``I-REQ``, ``E-REQ``. ``O`` means outside a
phrase; ``S-REQ`` is a single-token phrase. ``B-REQ``, ``I-REQ`` and
``E-REQ`` are the beginning, inside and end of a longer phrase.
The ``text`` field has no ``{{ }}`` markers: it is the text the model sees
at inference time. It is also used when checking predicted spans during
training, so it cannot currently be omitted. The original marked text remains
in the source ``.RULE`` file; this export does not alter it. ``rule_type`` is
the first set ScanCode license rule flag, for inspecting what kinds of rules
were exported. It is ``license_rule`` when no such flag is set; it is not
a model feature. Training currently expects this field to be a non-empty
string.

Splitting
---------

The current split groups records by license expression. Expressions with
fewer than 50 eligible records are considered rare. Their groups are sorted
by descending record count, then expression name; each entire group goes to
the split with the lowest current count relative to its target. Ties use
train, then val, then test. Targets for these rare records are 80/10/10,
but whole groups can make the actual proportions differ.

Expressions with 50 or more eligible records are split *per rule* using the
MD5 hash of the rule identifier: buckets 0–79 go to train, 80–89 to val,
and 90–99 to test. This is deterministic but does not guarantee exact
80/10/10 proportions. The validation and test splits therefore include
unseen rare expressions **and** rules from common expressions that may also
appear in training. They are not wholly unseen-license evaluations. The
threshold of 50 and the absence of rule-length or phrase-density balancing
still need analysis; this description does not establish that the split is
optimal.

Only eligible rules containing a marked required phrase are included.
Dedicated ``is_required_phrase`` rules, deprecated rules, false positives,
license clues, license introductions, and rules without a license expression
are excluded.
