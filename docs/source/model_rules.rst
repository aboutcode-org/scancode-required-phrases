Add model-predicted required phrases
====================================

Use the command after reviewing predictions from a validated final model:

.. code-block:: console

    add-model-required-phrases \
        --model model-output/final-model \
        --dry-run \
        --verbose

Install the ``inference`` extra before using this command. ``--model`` accepts
a local final-model directory or a Hugging Face repository. Remote models also
require their full commit hash through ``--model-revision``. The model must pass
the publication checks before inference starts.

The command skips rules that cannot receive generated required phrases and
rules that already contain required-phrase markers. It rejects phrase text
found more than once because ScanCode would mark every occurrence. The complete
rule update is checked in memory and each changed rule is written once.

Use ``--license-expression`` to process one expression and ``--limit`` for a
small review run. Remove ``--dry-run`` only after reviewing the predictions.
Rebuild the ScanCode license index after writing rules.
