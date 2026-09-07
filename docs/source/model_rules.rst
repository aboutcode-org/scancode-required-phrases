Add model-predicted required phrases
====================================

Use the command after reviewing predictions from a validated final model:

.. code-block:: console

    add-model-required-phrases \
        --model model-output/final-model \
        --dry-run \
        --verbose

``--model`` accepts a local final-model directory or a Hugging Face repository.
The model must pass the hardened publication checks before inference starts.

The command skips rules that cannot receive generated required phrases and
rules that already contain required-phrase markers. Each prediction must pass
ScanCode's candidate and locatability checks. Accepted phrases are applied in
memory and each changed rule is written once.

Use ``--license-expression`` to process one expression and ``--limit`` for a
small review run. Remove ``--dry-run`` only after reviewing the predictions.
Rebuild the ScanCode license index after writing rules.
