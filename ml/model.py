"""
ml/model.py

PyTorch models for handwriting recognition.

Phase 1: WordClassifier  - CNN + GlobalAvgPool -> word-level classification
Phase 2: CTCRecognizer   - CNN + BiLSTM + CTC  -> character-level decoding
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ── Phase 1: Word-level classifier ─────────────────────

class WordClassifier(nn.Module):
    """
    Classifies a variable-length accelerometer sequence into a word vocabulary.

    Input : (batch, time, 10)
    Output: (batch, num_words) logits
    """

    def __init__(self, num_features=10, num_words=30):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(num_features, 32, kernel_size=5, padding=2),
            ConvBlock(32, 64, kernel_size=5, padding=2),
            ConvBlock(64, 128, kernel_size=3, padding=1),
        )
        self.head = nn.Linear(128, num_words)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)  # (batch, features, time)
        x = self.encoder(x)    # (batch, 128, time)

        if lengths is not None:
            max_len = x.size(2)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()
            x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        else:
            x = x.mean(dim=2)

        return self.head(x)


# ── Phase 2: Character-level CTC recognizer ────────────

CHARS = " abcdefghijklmnopqrstuvwxyz"
BLANK_IDX = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {0: "", **{i + 1: c for i, c in enumerate(CHARS)}}
NUM_CLASSES = len(CHARS) + 1  # 28: blank + space + 26 letters


def encode_text(text):
    """Convert a string to character indices."""
    return [CHAR_TO_IDX[c] for c in text.lower() if c in CHAR_TO_IDX]


def decode_ctc(indices):
    """Greedy CTC decode: collapse repeats, remove blanks."""
    result = []
    prev = None
    for idx in indices:
        if idx != prev:
            if idx != BLANK_IDX:
                result.append(IDX_TO_CHAR.get(idx, ""))
            prev = idx
    return "".join(result)


class CTCRecognizer(nn.Module):
    """
    CNN + BiLSTM with CTC output for character-level recognition.

    Input : (batch, time, 10)
    Output: (time, batch, num_classes) log-probabilities
    """

    def __init__(self, num_features=10, hidden_size=128, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            ConvBlock(num_features, 64, kernel_size=5, padding=2),
            ConvBlock(64, 128, kernel_size=5, padding=2),
            ConvBlock(128, 128, kernel_size=3, padding=1),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        x, _ = self.lstm(x)
        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        logits = self.fc(x)
        log_probs = logits.log_softmax(dim=2).transpose(0, 1)

        output_lengths = lengths if lengths is not None else torch.full(
            (x.size(0),), x.size(1), dtype=torch.long
        )
        return log_probs, output_lengths
