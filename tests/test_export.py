# -*- coding: utf-8 -*-
#
# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

np = pytest.importorskip("numpy")

from scancode_required_phrases.export import sha256
from scancode_required_phrases.export import viterbi_decode


def test_viterbi_with_zero_transitions_is_argmax():
    emissions = np.array(
        [
            [0.1, 0.9, 0.0, 0.0, 0.0],
            [0.7, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.9],
        ]
    )
    transitions = np.zeros((5, 5))
    edges = np.zeros(5)

    assert viterbi_decode(emissions, edges, transitions, edges) == [1, 0, 4]


def test_viterbi_obeys_transition_scores():
    emissions = np.array([[0.0, 1.0], [1.0, 0.0]])
    transitions = np.array([[0.0, 0.0], [-100.0, 0.0]])
    edges = np.zeros(2)

    path = viterbi_decode(emissions, edges, transitions, edges)

    assert path[0] == path[1]


def test_sha256_is_stable(tmp_path):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"model")

    assert sha256(path) == "9372c470eeadd5ecd9c3c74c2b3cb633f8e2f2fad799250a0f70d652b6b825e4"
