"""Utility functions for multiple-choice molecular QA benchmarks."""

import json
import re
from typing import Dict


def parse_options(prompt: str) -> Dict[str, str]:
    """Extract the mapping between option letters and their textual values."""

    options: Dict[str, str] = {}
    for letter, value in re.findall(r"([A-D])\.\s*([^\n]+)", prompt):
        options[letter] = value.strip()
    return options


def normalize_label(label: str):
    return label.strip()


def is_number(s: str):
    try:
        float(s)
        return True
    except ValueError:
        return False


def compute_accuracy(jsonl_file: str) -> float:
    total = 0
    correct = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            prompt = data["prompt"]
            pred = data["predict"].strip()
            label = normalize_label(data["label"])

            options = parse_options(prompt)
            gold_letter = label
            gold_value = options[gold_letter]

            # Evaluate the prediction.
            if pred in options:  # Option letter provided directly
                if pred == gold_letter:
                    correct += 1
            elif is_number(pred):  # Numeric prediction
                try:
                    pred_val = round(float(pred), 2)
                    gold_val = round(float(gold_value), 2)
                    if pred_val == gold_val:
                        correct += 1
                except Exception:
                    pass
            else:
                # Handle answers embedded in natural language sentences.
                m = re.search(r"\b([A-D])\b", pred)
                if m and m.group(1) == gold_letter:
                    correct += 1

            total += 1

    acc = correct / total if total > 0 else 0
    return acc


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} file.jsonl")
        exit(1)

    acc = compute_accuracy(sys.argv[1])
    print(f"Accuracy: {acc:.4f}")
