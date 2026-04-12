"""
pennference.py

End-to-end inference pipeline: pen -> segmentation -> classification -> terminal.

Reads the accelerometer stream, uses the SegmentationTCN to detect word
boundaries, classifies each extracted segment with the WordClassifier,
and prints confident predictions to the terminal.

Usage:
    python3 pennference.py
    python3 pennference.py --confidence 0.6
    python3 pennference.py --verbose          # show all predictions + writing state
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from serial_utils import find_arduino_port, open_serial, parse_line
from training.model import SegmentationTCN, WordClassifier
from training.data_pipeline import trim_idle, compute_features

# Default model paths
SEGMENTER_PATH = Path(__file__).resolve().parent / "segmenter.pt"
CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "word_classifier_best.pt"

DEFAULT_CONFIDENCE = 0.5


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Segmenter: detects word boundaries in the accelerometer stream
# ---------------------------------------------------------------------------

class WordSegmenter:
    """
    Wraps SegmentationTCN with Schmitt-trigger hysteresis to extract
    word segments from a continuous accelerometer stream.

    Feed samples one at a time. Returns the extracted word (as [[x,y,z], ...])
    when a word boundary is detected, None otherwise.
    """

    BUFFER_SIZE = 256
    THRESHOLD_HI = 0.55   # enter "writing" when P(writing) exceeds this
    THRESHOLD_LO = 0.40   # exit "writing" when P(writing) drops below this
    MIN_WORD_SAMPLES = 25  # ~0.5s at 50Hz
    MIN_GAP_SAMPLES = 8   # ~0.16s at 50Hz

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

        from collections import deque
        self.buffer = deque(maxlen=self.BUFFER_SIZE)
        self.state = "idle"       # idle | writing
        self.writing_count = 0    # consecutive frames above HI threshold
        self.gap_count = 0        # consecutive frames below LO threshold
        self.word_samples = []    # accumulated [x,y,z] during current word

    def feed(self, x, y, z):
        """Feed one accelerometer sample.

        Returns:
            None                - no event
            ("writing",)        - word started
            ("word", samples)   - word ended, samples is [[x,y,z], ...]
        """
        self.buffer.append((x, y, z))

        if len(self.buffer) < 30:
            return None

        # Run segmenter on the buffer
        arr = np.array(list(self.buffer), dtype=np.float32)
        with torch.no_grad():
            inp = torch.from_numpy(arr).unsqueeze(0).to(self.device)
            logits = self.model(inp)
            prob = torch.sigmoid(logits[0, -1]).item()

        if self.state == "idle":
            if prob > self.THRESHOLD_HI:
                self.writing_count += 1
                if self.writing_count >= 3:
                    self.state = "writing"
                    self.gap_count = 0
                    self.word_samples = []
                    return ("writing",)
            else:
                self.writing_count = 0

        elif self.state == "writing":
            self.word_samples.append([x, y, z])

            if prob < self.THRESHOLD_LO:
                self.gap_count += 1
            else:
                self.gap_count = 0

            if (self.gap_count >= self.MIN_GAP_SAMPLES
                    and len(self.word_samples) >= self.MIN_WORD_SAMPLES):
                # Trim trailing gap frames from the word
                trim = min(self.MIN_GAP_SAMPLES, len(self.word_samples))
                samples = self.word_samples[:-trim]
                self.word_samples = []
                self.state = "idle"
                self.writing_count = 0
                self.gap_count = 0
                return ("word", samples)

        return None


# ---------------------------------------------------------------------------
# Classifier: maps an extracted word segment to a vocabulary word
# ---------------------------------------------------------------------------

def classify_word(model, idx_to_word, samples_xyz, device):
    """Classify an extracted word segment.

    Args:
        model: trained WordClassifier
        idx_to_word: {int: str} vocabulary mapping
        samples_xyz: [[x, y, z], ...] raw accelerometer data for one word
        device: torch device

    Returns:
        (predicted_word, confidence, top3) where top3 is [(word, prob), ...]
    """
    trimmed = trim_idle(samples_xyz)
    features = compute_features(trimmed)
    features_t = torch.from_numpy(features).unsqueeze(0).to(device)  # (1, T, 10)
    lengths = torch.tensor([features_t.size(1)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(features_t, lengths)
        probs = torch.softmax(logits, dim=1)[0]  # (num_words,)

    # Top prediction
    confidence, pred_idx = probs.max(dim=0)
    word = idx_to_word[pred_idx.item()]

    # Top 3 for verbose output
    top_k = torch.topk(probs, min(3, len(probs)))
    top3 = [(idx_to_word[i.item()], p.item()) for i, p in zip(top_k.indices, top_k.values)]

    return word, confidence.item(), top3


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ThePenn: write with the pen, see words in the terminal")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port (auto-detect if omitted)")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                        help=f"Minimum confidence to accept a prediction (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--segmenter", type=str, default=str(SEGMENTER_PATH),
                        help="Path to segmenter .pt file")
    parser.add_argument("--classifier", type=str, default=str(CLASSIFIER_PATH),
                        help="Path to classifier .pt file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show all predictions with confidence scores")
    args = parser.parse_args()

    # --- Validate model files ---
    if not Path(args.segmenter).exists():
        print(f"Segmenter not found: {args.segmenter}")
        print("Run: python3 -m training.train_segmenter")
        sys.exit(1)
    if not Path(args.classifier).exists():
        print(f"Classifier not found: {args.classifier}")
        print("Run: python3 -m training.train")
        sys.exit(1)

    device = get_device()

    # --- Load segmenter ---
    seg_model = SegmentationTCN(
        in_channels=3, hidden=64, kernel_size=3,
        num_blocks=5, dropout=0.15,
    ).to(device)
    seg_model.load_state_dict(
        torch.load(args.segmenter, weights_only=True, map_location=device))
    seg_model.eval()

    # --- Load classifier ---
    ckpt = torch.load(args.classifier, weights_only=False, map_location=device)
    idx_to_word = ckpt["idx_to_word"]
    word_to_idx = ckpt["word_to_idx"]
    num_words = ckpt["num_words"]

    cls_model = WordClassifier(num_features=10, num_words=num_words).to(device)
    cls_model.load_state_dict(ckpt["model_state_dict"])
    cls_model.eval()

    vocab = sorted(word_to_idx.keys())
    print(f"Device:     {device}")
    print(f"Vocabulary: {num_words} words")
    print(f"  {', '.join(vocab)}")
    print(f"Confidence: {args.confidence:.0%}")
    print()

    # --- Connect to Arduino ---
    port = args.port or find_arduino_port()
    if not port:
        print("No Arduino found. Connect the device and try again.")
        print("Available ports:")
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            print(f"  {p.device}  {p.description}")
        sys.exit(1)

    print(f"Connecting to {port}...")
    ser = open_serial(port)
    print("Connected. Start writing!")
    print()
    print("=" * 50)
    print()

    segmenter = WordSegmenter(seg_model, device)
    word_count = 0
    output_words = []

    def log(msg):
        ts = time.strftime("%H:%M:%S")
        print(f"  {ts}  {msg}")

    try:
        while True:
            raw = ser.readline()
            if not raw:
                continue
            parsed = parse_line(raw)
            if parsed is None:
                continue

            x, y, z = parsed
            event = segmenter.feed(x, y, z)

            if event is None:
                continue

            if event[0] == "writing":
                log("WORD STARTED")

            elif event[0] == "word":
                samples = event[1]
                log(f"WORD DETECTED  ({len(samples)} samples)")
                log("CLASSIFYING...")

                word, confidence, top3 = classify_word(
                    cls_model, idx_to_word, samples, device)

                top3_str = "  ".join(f"{w} {p:.0%}" for w, p in top3)
                log(f"  top 3:  {top3_str}")

                if confidence >= args.confidence:
                    word_count += 1
                    output_words.append(word)
                    log(f"WORD IS: {word}  ({confidence:.0%})")
                else:
                    log(f"LOW CONFIDENCE: {word}  ({confidence:.0%}) -- skipped")

                log(f"TOTAL WORDS: {word_count}")
                print(f"  >>> {' '.join(output_words)}")
                print()

    except KeyboardInterrupt:
        print()
        print("=" * 50)
        print(f"Session: {word_count} words recognized")
        if output_words:
            print(f"Output: {' '.join(output_words)}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()
