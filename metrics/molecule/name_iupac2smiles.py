import json


def extract_smiles(text: str) -> str:
    """Extract the canonical SMILES substring from phrases such as ``"The SMILES of the molecule is ..."``."""

    return text.split("is", 1)[-1].strip()


def compute_metrics(jsonl_file):
    total_tokens = 0
    correct_tokens = 0
    total_seqs = 0
    correct_seqs = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            pred = extract_smiles(data["predict"])
            gold = extract_smiles(data["label"])

            # Token-level comparison is performed character by character.
            min_len = min(len(pred), len(gold))
            for i in range(min_len):
                if pred[i] == gold[i]:
                    correct_tokens += 1
            total_tokens += max(len(pred), len(gold))

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

    token_acc, seq_acc = compute_metrics(sys.argv[1])
    print(f"Token Accuracy: {token_acc:.4f}")
    print(f"Seq Accuracy:   {seq_acc:.4f}")
