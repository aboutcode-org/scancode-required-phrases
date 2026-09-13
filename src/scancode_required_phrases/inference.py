# Copyright (c) nexB Inc. and others. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only required-phrase prediction from a validated final model."""

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from licensedcode.tokenize import required_phrase_splitter

from scancode_required_phrases.training import encode_complete_words
from scancode_required_phrases.training import extract_spans
from scancode_required_phrases.training import first_subword_positions
from scancode_required_phrases.training import ID2LABEL
from scancode_required_phrases.training import load_final_model


@dataclass(frozen=True)
class PhrasePrediction:
    """One predicted required-phrase candidate."""

    text: str
    start_word: int
    end_word: int
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """Read-only predictions and tokenization details for one rule text."""

    words: tuple[str, ...]
    phrases: tuple[PhrasePrediction, ...]
    truncated: bool


def words_from_text(text):
    """Tokenize rule text exactly as the training dataset does."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return required_phrase_splitter(unicodedata.normalize("NFKC", text))


def span_confidence(crf, word_emissions, tags, mask, free, span):
    """Return the CRF probability mass agreeing with one decoded span."""
    start, end = span
    pinned = word_emissions.clone()
    floor = float(word_emissions.min()) - 10000.0

    for position in range(start, end + 1):
        label = int(tags[0, position])
        keep = float(pinned[0, position, label])
        pinned[0, position] = floor
        pinned[0, position, label] = keep

    constrained = crf(pinned, tags, mask=mask, reduction="none")
    confidence = float((free - constrained).detach().exp())
    return min(max(confidence, 0.0), 1.0)


class RequiredPhrasePredictor:
    """Run a validated hardened model without changing ScanCode rules."""

    def __init__(self, model, tokenizer, max_length):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    @classmethod
    def from_model_dir(cls, model_dir):
        """Validate and load a local final-model directory."""
        model_dir = Path(model_dir)
        model, tokenizer = load_final_model(model_dir)
        config = json.loads((model_dir / "train_config.json").read_text(encoding="utf-8"))
        return cls(model=model, tokenizer=tokenizer, max_length=config["max_length"])

    def predict(self, text):
        """Return candidate spans for text without mutating any rule or file."""
        import torch

        words = words_from_text(text)
        if not words:
            return PredictionResult(words=(), phrases=(), truncated=False)

        encoding, truncated = encode_complete_words(
            tokens=words,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        positions = first_subword_positions(encoding.word_ids())
        if not positions:
            return PredictionResult(words=tuple(words), phrases=(), truncated=False)

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([encoding["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [encoding["attention_mask"]],
            dtype=torch.long,
            device=device,
        )

        with torch.inference_mode():
            emissions = self.model.emissions(input_ids, attention_mask)
            word_emissions = emissions[:, positions].float()
            mask = torch.ones(
                word_emissions.shape[:2],
                dtype=torch.bool,
                device=word_emissions.device,
            )
            decoded = self.model.crf.decode(word_emissions, mask=mask)[0]
            tags = torch.tensor([decoded], device=word_emissions.device)
            free = self.model.crf(word_emissions, tags, mask=mask, reduction="none")

            labels = [ID2LABEL[int(label)] for label in decoded]
            predictions = []
            for start, end in extract_spans(labels):
                predictions.append(
                    PhrasePrediction(
                        text=" ".join(words[start : end + 1]),
                        start_word=start,
                        end_word=end,
                        confidence=span_confidence(
                            self.model.crf,
                            word_emissions,
                            tags,
                            mask,
                            free,
                            (start, end),
                        ),
                    )
                )

        predictions.sort(key=lambda item: (item.start_word, item.end_word))
        return PredictionResult(
            words=tuple(words),
            phrases=tuple(predictions),
            truncated=truncated,
        )
