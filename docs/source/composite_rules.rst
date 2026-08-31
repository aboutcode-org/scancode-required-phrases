Update composite rules
======================

The ``add-composite-required-phrases`` command adds existing required phrases
to ScanCode rules with multi-key license expressions.

Run the command
---------------

Preview changes without saving rules:

.. code-block:: console

    add-composite-required-phrases --dry-run --verbose

Use ``--license-expression`` to process one expression. Use
``--write-phrase-source`` to record the source required phrase rules in updated
rules.

Matching
--------

Candidates come from existing ``is_required_phrase`` rules with one
non-generic license key. A composite rule is updated only when every relevant
key has a non-overlapping phrase match. Existing required phrase markers are
kept, and filenames and URLs are not changed.

Writing and validation
----------------------

The command operates on the installed ScanCode rules directory. Without
``--dry-run``, each changed rule is written once after all required phrases are
added. Use ``--validate`` to validate the rules and licenses and ``--reindex``
to rebuild the cached license index after updating.
