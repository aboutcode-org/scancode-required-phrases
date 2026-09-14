Model prediction and rule preparation
=====================================

Install the ``inference`` extra before loading a final model. A local model must
pass the final-model publication checks. A remote Hugging Face model also
requires its full commit hash.

The ``scancode_required_phrases.model_rules`` module provides reusable functions
to:

- load eligible rules from one file, a directory, or installed ScanCode data;
- return model predictions with their ScanCode validation result;
- prepare a complete rule update without mutating the original rule;
- serialize and atomically write a rule to its exact source path.

Phrase text found more than once is rejected because ScanCode's mutation helper
would mark every occurrence. A complete phrase set is prepared before any file
is written. The stacked review workflow provides the user-facing command and
human approval process.
