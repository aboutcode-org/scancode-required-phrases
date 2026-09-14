Train and export a required phrase model
=========================================

The training command fine-tunes the reviewed DeBERTa word-input classifier with
BIOES labels and an optional constrained word-level CRF. It consumes the hybrid
JSONL splits produced by ``build-required-phrases-dataset`` without moving,
deduplicating, retokenizing, relabeling, or repairing records.

Installation
------------

Install training support with ``python -m pip install ".[training]"``. Add the
``training-8bit`` extra only when using ``adamw-8bit``. ONNX support remains
optional and is installed separately with ``.[onnx]``. The training extra pins
Transformers 4.57.3 and Hugging Face Hub 0.36.2, the versions validated by the
single-T4 Kaggle smoke run.

Training contract
-----------------

A run requires a full immutable model-host commit in ``--model-revision`` and a
new or empty ``--output-dir``. For the current DeBERTa base model, run:

.. code-block:: console

    train-required-phrase-model --data-dir dataset-output --output-dir model-output \
        --model-revision 64a8c8eab3e352a784c658aef62be1662607476f \
        --optimizer adamw-8bit

Resume is unsupported. Every non-empty line in
all three splits is validated before tokenizer loading or ``--limit`` selection.
Records must use exact field types, contain a positive valid BIOES sequence, and
have unique identifiers across all splits. Exact token and label duplicates are
reported but never removed.

Alignment uses the dataset's existing words. Full untruncated fast-tokenizer
coverage proves the complete retained word prefix. Coverage gaps, zero-subword
words, partial words, and truncation that omits any non-``O`` label reject the
record. ``--limit`` is applied only after complete validation and alignment.

The run records versioned raw-byte (H0), validated-record (H1), and effective
example (H2) hashes. ``run_manifest.json`` is atomically replaced through
pre-run, failure, or success state and contains source/runtime provenance from
runtime APIs, never an environment dump.

Final model
-----------

Selection remains based only on best validation strict span F1. The selected
checkpoint, selected in-memory model, and staged model must have exactly equal
states. ``final-model.tmp`` is then reloaded using only its local configuration,
tokenizer, and full weights; ordered validation predictions, labels, invalid
path count, and metrics must match exactly. The stage is atomically promoted to
``final-model`` and ``SUCCESS.json`` is written last. A model is publishable
only while that marker and every recorded file hash validate.

Strict metrics reject malformed gold paths and malformed CRF predictions.
Malformed non-CRF predictions are not repaired: they produce no predicted spans,
count all gold spans as false negatives, and increment ``invalid_paths``.

ISR retains its existing name but measures predicted-phrase locatability only.
It does not exercise injection gates or rule mutation and is not evidence of
injection success.

Read-only prediction
--------------------

``RequiredPhrasePredictor`` loads only a final model that passes the publication
checks. Its ``predict()`` method uses the same ScanCode tokenization as the
training dataset and returns phrase text, word offsets, a model score, and whether
the input was truncated. It does not change rules or write files.

.. code-block:: python

    from scancode_required_phrases.inference import RequiredPhrasePredictor

    predictor = RequiredPhrasePredictor.from_model_dir("model-output/final-model")
    result = predictor.predict("Permission is hereby granted ...")

Treat every prediction as a candidate requiring human review.

Export
------

``export-required-phrase-model`` validates a publishable local final model and
exports constrained CRF matrices by default:

.. code-block:: console

    export-required-phrase-model --model-dir model-output/final-model \
        --output-dir model-export

Run a separate export with ``--operation onnx`` and a different new output
directory for optional ONNX emissions; missing ONNX packages cannot disable CRF
export, training, finalization, or offline reload.

Generated datasets, reports, manifests, checkpoints, final models, matrix
files, ONNX files, smoke outputs, and caches must remain outside commits and the
pull request.
