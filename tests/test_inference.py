# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

os.environ.setdefault("USE_TF", "0")

torch = pytest.importorskip("torch")
pytest.importorskip("torchcrf")
pytest.importorskip("transformers")

from scancode_required_phrases import inference
from scancode_required_phrases.inference import RequiredPhrasePredictor
from scancode_required_phrases.inference import words_from_text
from scancode_required_phrases.model import ConstrainedCRF
from scancode_required_phrases.training import LABEL2ID
from scancode_required_phrases.training import LABELS


class FakeEncoding(dict):

    def __init__(self, word_ids):
        super().__init__(
            input_ids=[0] * len(word_ids),
            attention_mask=[1] * len(word_ids),
        )
        self._word_ids = word_ids

    def word_ids(self):
        return self._word_ids


class FakeTokenizer:

    is_fast = True
    all_special_ids = [0]
    vocab_size = 100

    def __init__(self, subwords=None):
        self.subwords = subwords or {}

    def __call__(self, words, truncation, max_length=None, **kwargs):
        word_ids = [None]
        for index, word in enumerate(words):
            word_ids.extend([index] * self.subwords.get(word, 1))
        word_ids.append(None)
        if truncation and len(word_ids) > max_length:
            word_ids = word_ids[: max_length - 1] + [None]
        return FakeEncoding(word_ids)


class StubTagger(torch.nn.Module):

    def __init__(self, tagged):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.tagged = tagged
        self.crf = ConstrainedCRF(len(LABELS), batch_first=True)
        with torch.no_grad():
            for parameter in self.crf.parameters():
                parameter.zero_()

    def emissions(self, input_ids, attention_mask):
        scores = torch.zeros((1, input_ids.shape[1], len(LABELS)))
        scores[:, :, LABEL2ID["O"]] = 4.0
        for word, label in self.tagged.items():
            scores[0, word + 1] = 0.0
            scores[0, word + 1, LABEL2ID[label]] = 9.0
        return scores


def test_words_match_dataset_tokenization():
    assert words_from_text("Apache-2.0 License") == ["Apache", "2", "0", "License"]
    assert words_from_text("a\ufb01x\r\nnotice") == ["afix", "notice"]


def test_loads_predictor_from_validated_model(tmp_path, monkeypatch):
    model = StubTagger({})
    tokenizer = FakeTokenizer()
    (tmp_path / "train_config.json").write_text(
        json.dumps({"max_length": 384}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        inference,
        "load_final_model",
        lambda model_dir: (model, tokenizer),
    )

    predictor = RequiredPhrasePredictor.from_model_dir(tmp_path)

    assert predictor.model is model
    assert predictor.tokenizer is tokenizer
    assert predictor.max_length == 384


def test_predicts_bioes_phrase_without_mutation():
    text = "Granted under the MIT License to everyone"
    predictor = RequiredPhrasePredictor(
        model=StubTagger({3: "B-REQ", 4: "E-REQ"}),
        tokenizer=FakeTokenizer(),
        max_length=512,
    )

    result = predictor.predict(text)

    assert text == "Granted under the MIT License to everyone"
    assert [phrase.text for phrase in result.phrases] == ["MIT License"]
    assert result.phrases[0].start_word == 3
    assert result.phrases[0].end_word == 4
    assert 0.0 <= result.phrases[0].confidence <= 1.0
    assert not result.truncated


def test_predicts_single_word_phrase():
    predictor = RequiredPhrasePredictor(
        model=StubTagger({2: "S-REQ"}),
        tokenizer=FakeTokenizer(),
        max_length=512,
    )

    result = predictor.predict("one two MIT four")

    assert [phrase.text for phrase in result.phrases] == ["MIT"]


def test_keeps_valid_phrase_at_truncation_boundary():
    predictor = RequiredPhrasePredictor(
        model=StubTagger({2: "S-REQ"}),
        tokenizer=FakeTokenizer(),
        max_length=5,
    )

    result = predictor.predict("one two three four five six")

    assert result.truncated
    assert [phrase.text for phrase in result.phrases] == ["three"]


def test_empty_text_does_not_run_model():
    predictor = RequiredPhrasePredictor(
        model=StubTagger({}),
        tokenizer=FakeTokenizer(),
        max_length=512,
    )

    assert predictor.predict("").phrases == ()
