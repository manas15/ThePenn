"""
training/model.py

PyTorch model definitions for ghost-writer handwriting recognition.

Phase 1: WordClassifier       — CNN + GlobalAvgPool → word-level classification
Phase 2: CTCRecognizer        — CNN + BiLSTM + CTC  → character-level decoding
Segmentation: SegmentationTCN — causal TCN → per-sample writing/gap detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=5, padding=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Phase 1: Word-level classifier
# ---------------------------------------------------------------------------

class WordClassifier(nn.Module):
    """
    Classifies a variable-length accelerometer sequence into a fixed word vocabulary.

    Input : (batch, time, features)   — features = 10 by default
    Output: (batch, num_words)        — logits over word vocabulary
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
        """
        Args:
            x: (batch, time, features) — padded input
            lengths: (batch,) — true lengths before padding (optional)
        Returns:
            logits: (batch, num_words)
        """
        # Conv1d expects (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.encoder(x)  # (batch, 128, time)

        # Masked global average pooling
        if lengths is not None:
            # Create mask: (batch, 1, time)
            max_len = x.size(2)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()  # (batch, 1, time)
            x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        else:
            x = x.mean(dim=2)  # (batch, 128)

        return self.head(x)


# ---------------------------------------------------------------------------
# Word validator: binary classifier — is this a word or not?
# ---------------------------------------------------------------------------

class WordValidator(nn.Module):
    """
    Binary classifier: is this accelerometer segment a real word?

    Same CNN backbone as WordClassifier but with a single sigmoid output.

    Input : (batch, time, features)   — features = 10 by default
    Output: (batch,)                  — logit, apply sigmoid for P(word)
    """

    def __init__(self, num_features=10):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(num_features, 32, kernel_size=5, padding=2),
            ConvBlock(32, 64, kernel_size=5, padding=2),
            ConvBlock(64, 128, kernel_size=3, padding=1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)
        x = self.encoder(x)

        if lengths is not None:
            max_len = x.size(2)
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(1).float()
            x = (x * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)
        else:
            x = x.mean(dim=2)

        return self.head(x).squeeze(1)


# ---------------------------------------------------------------------------
# Phase 2: Character-level CTC recognizer
# ---------------------------------------------------------------------------

# Character set: 26 lowercase letters + space + CTC blank (index 0)
CHARS = " abcdefghijklmnopqrstuvwxyz"
BLANK_IDX = 0
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}  # blank=0, space=1, a=2, ...
IDX_TO_CHAR = {0: "", **{i + 1: c for i, c in enumerate(CHARS)}}
NUM_CLASSES = len(CHARS) + 1  # 28 (blank + space + 26 letters)


def encode_text(text):
    """Convert a string to a list of character indices."""
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
    CNN + BiLSTM encoder with CTC output for character-level recognition.

    Input : (batch, time, features)
    Output: (batch, time, num_classes) — log-probabilities for CTC loss
    """

    def __init__(self, num_features=10, hidden_size=128, num_layers=2,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
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
        """
        Args:
            x: (batch, time, features) — padded input
            lengths: (batch,) — true lengths before padding
        Returns:
            log_probs: (time, batch, num_classes) — CTC format
            output_lengths: (batch,) — same as input lengths (no downsampling)
        """
        # CNN encoder
        x = x.transpose(1, 2)          # (batch, features, time)
        x = self.encoder_cnn(x)        # (batch, 128, time)
        x = x.transpose(1, 2)          # (batch, time, 128)

        # BiLSTM
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        x, _ = self.lstm(x)
        if lengths is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Project to character classes
        logits = self.fc(x)                          # (batch, time, num_classes)
        log_probs = logits.log_softmax(dim=2)        # (batch, time, num_classes)
        log_probs = log_probs.transpose(0, 1)        # (time, batch, num_classes) — CTC expects this

        output_lengths = lengths if lengths is not None else torch.full(
            (x.size(0),), x.size(1), dtype=torch.long
        )
        return log_probs, output_lengths


# ---------------------------------------------------------------------------
# Segmentation: TCN for writing/gap detection
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Conv1d with left-only padding so output[t] depends only on input[<=t]."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCNBlock(nn.Module):
    """Two causal convolutions with residual connection."""

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        r = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop(x)
        return x + r


class SegmentationTCN(nn.Module):
    """
    Causal temporal convolutional network for binary segmentation.

    Takes a continuous accelerometer stream and outputs a per-sample
    probability that the pen is currently writing (vs. in a between-word gap).

    Receptive field with default settings (5 blocks, kernel 3):
      sum of d*(K-1)*2 for d in [1,2,4,8,16] = 2*(1+2+4+8+16)*2 = 124 samples
      ≈ 2.5s at 50Hz — covers a full word.

    Input : (batch, time, in_channels)
    Output: (batch, time)  — raw logits, apply sigmoid for probabilities
    """

    def __init__(self, in_channels=3, hidden=64, kernel_size=3,
                 num_blocks=5, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.output_proj = nn.Conv1d(hidden, 1, 1)

    def forward(self, x, lengths=None):
        x = x.transpose(1, 2)              # (batch, channels, time)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_proj(x).squeeze(1)  # (batch, time)
        return x
