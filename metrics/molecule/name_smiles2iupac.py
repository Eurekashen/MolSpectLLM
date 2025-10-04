import json


def tokenize(text: str, by_char: bool = True):
    """Tokenize IUPAC name by character (default) or by whitespace."""
    return list(text.strip()) if by_char else text.strip().split()


def compute_metrics(jsonl_file, by_char=True):
    total_tokens = 0
    correct_tokens = 0
    total_seqs = 0
    correct_seqs = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            pred = data["predict"].strip()
            gold = data["label"].strip()

            # token-level
            pred_tokens = tokenize(pred, by_char)
            gold_tokens = tokenize(gold, by_char)

            min_len = min(len(pred_tokens), len(gold_tokens))
            for i in range(min_len):
                if pred_tokens[i] == gold_tokens[i]:
                    correct_tokens += 1
            total_tokens += max(len(pred_tokens), len(gold_tokens))

            # seq-level
            if pred == gold:
                correct_seqs += 1
            total_seqs += 1

    token_acc = correct_tokens / total_tokens if total_tokens > 0 else 0
    seq_acc = correct_seqs / total_seqs if total_seqs > 0 else 0
    return token_acc, seq_acc


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} file.jsonl")
        exit(1)

    token_acc, seq_acc = compute_metrics(sys.argv[1], by_char=True)
    print(f"Token Accuracy: {token_acc:.4f}")
    print(f"Seq Accuracy:   {seq_acc:.4f}")
