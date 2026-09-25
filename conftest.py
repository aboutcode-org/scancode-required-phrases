# -*- coding: utf-8 -*-
#
# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest collection settings for optional training dependencies."""

import importlib.util


training_dependencies = ("torch", "torchcrf", "transformers")
if any(importlib.util.find_spec(name) is None for name in training_dependencies):
    collect_ignore = ["src/scancode_required_phrases/model.py"]
else:
    collect_ignore = []
