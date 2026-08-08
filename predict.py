"""Classify one text input using local model and vocabulary artifacts only."""
import argparse
import pickle
from pathlib import Path

import torch

from transformer_encoder.lora import apply_lora
from transformer_encoder.model import TransformerClassifier


def load_vocabulary(path: Path) -> dict[str, int]:
    with path.open("rb") as file:
        vocabulary = pickle.load(file)
    return vocabulary["stoi"]


def encode(text: str, stoi: dict[str, int], max_length: int) -> list[int]:
    unk_id, pad_id = stoi["<unk>"], stoi["<pad>"]
    token_ids = [stoi.get(token, unk_id) for token in text.lower().split()]
    return (token_ids[:max_length] + [pad_id] * max_length)[:max_length]


def main():
    parser = argparse.ArgumentParser(description="Predict an AG News topic with the LoRA checkpoint.")
    parser.add_argument("--text", required=True, help="News headline or article text to classify")
    parser.add_argument("--method", choices=["full", "lora"], default="lora", help="Checkpoint type to load")
    parser.add_argument("--checkpoint", help="Override the checkpoint path")
    parser.add_argument("--vocab", help="Path to vocab.pkl; defaults to the checkpoint directory")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint or f"outputs/{args.method}_model.pt")
    vocab_path = Path(args.vocab) if args.vocab else checkpoint_path.parent / "vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Run train.py once to create vocab.pkl.")
    stoi = load_vocabulary(vocab_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        max_length, num_classes = checkpoint["max_length"], checkpoint["num_classes"]
        labels, config, state_dict = checkpoint["labels"], checkpoint["model_config"], checkpoint["state_dict"]
    else:
        # Compatibility with older raw state-dict checkpoints; no dataset access is required.
        max_length, num_classes = 64, 4
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        config, state_dict = {"d_model": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 256}, checkpoint

    model = TransformerClassifier(len(stoi), max_length=max_length, num_classes=num_classes, **config)
    method = checkpoint.get("method", args.method) if isinstance(checkpoint, dict) else args.method
    if method == "lora":
        apply_lora(model)
    model.load_state_dict(state_dict); model.eval()
    ids = torch.tensor(encode(args.text, stoi, max_length)).unsqueeze(0)
    with torch.no_grad(): probabilities = torch.softmax(model(ids), dim=1)[0]
    index = probabilities.argmax().item()
    print(f"Prediction: {labels[index]}")
    print(f"Confidence: {probabilities[index].item() * 100:.2f}%")
    print("All probabilities: " + ", ".join(f"{label}={probability:.3f}" for label, probability in zip(labels, probabilities.tolist())))


if __name__ == "__main__":
    main()
