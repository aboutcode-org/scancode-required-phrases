Train and export a required phrase model
=========================================

The training command fine-tunes a DeBERTa token classifier with BIOES labels
and an optional word-level CRF. It reads the JSONL files produced by
``build-required-phrases-dataset``.

Installation
------------

Install the training dependencies before running the command. The 8-bit
optimizer is a separate extra because it is only needed on constrained GPUs.

.. code-block:: console

    python -m pip install ".[training,training-8bit]"

Install the ONNX dependencies when an export is needed:

.. code-block:: console

    python -m pip install ".[onnx]"

Training
--------

.. code-block:: console

    train-required-phrase-model \
        --data-dir dataset-output \
        --output-dir model-output \
        --optimizer adamw-8bit

The default configuration uses FP32 and standard AdamW. The 8-bit optimizer is
recommended for DeBERTa-large on a 16 GB GPU. Test-set evaluation is disabled by
default so repeated experiments select models using validation data only. Add
``--evaluate-test`` only for the final selected run.

The command validates the dataset, skips examples whose required phrase is cut
by tokenizer truncation, and saves the checkpoint with the best validation
strict span F1. The output directory must be empty unless ``--resume`` is used.
It writes ``train_config.json`` for inference compatibility and
``run_manifest.json`` for reproducibility.

Export
------

.. code-block:: console

    export-required-phrase-model \
        --model-dir model-output \
        --output-dir onnx-output

The export command strictly loads the saved model, exports the DeBERTa emissions
to ONNX, stores the CRF transitions separately, and verifies that NumPy and
PyTorch Viterbi decoding agree.

Generated datasets, checkpoints, model files, and export artifacts must not be
committed to this repository.
