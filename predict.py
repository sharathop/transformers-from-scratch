"""Classify one text input using a LoRA-trained AG News Transformer checkpoint."""
import argparse

import torch

from transformer_encoder.data import Vocabulary, make_splits
from transformer_encoder.lora import apply_lora
from transformer_encoder.model import TransformerClassifier


def vocabulary_from_tokens(tokens):
    vocab = Vocabulary([], min_frequency=1)
    vocab.itos = tokens
    vocab.stoi = {word: index for index, word in enumerate(tokens)}
    return vocab


def main():
    parser = argparse.ArgumentParser(description="Predict an AG News topic with the LoRA checkpoint.")
    parser.add_argument("--text", required=True, help="News headline or article text to classify")
    parser.add_argument("--method", choices=["full", "lora"], default="lora", help="Checkpoint type to load")
    parser.add_argument("--checkpoint", help="Override the checkpoint path")
    parser.add_argument("--train-samples", type=int, default=30_000, help="Used only for old state-dict checkpoints")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or f"outputs/{args.method}_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        vocab = vocabulary_from_tokens(checkpoint["vocab_itos"])
        max_length, num_classes = checkpoint["max_length"], checkpoint["num_classes"]
        labels, config, state_dict = checkpoint["labels"], checkpoint["model_config"], checkpoint["state_dict"]
    else:
        # Compatibility with checkpoints saved before prediction metadata was added.
        print("Old checkpoint detected: rebuilding the deterministic AG News vocabulary once...")
        train_set, _, vocab, num_classes, labels = make_splits("ag_news", train_samples=args.train_samples, test_samples=7_600, max_length=64, seed=7)
        max_length, config, state_dict = train_set.max_length, {"d_model": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 256}, checkpoint

    model = TransformerClassifier(len(vocab), max_length=max_length, num_classes=num_classes, **config)
    method = checkpoint.get("method", args.method) if isinstance(checkpoint, dict) else args.method
    if method == "lora":
        apply_lora(model)
    model.load_state_dict(state_dict); model.eval()
    ids = torch.tensor(vocab.encode(args.text, max_length)).unsqueeze(0)
    with torch.no_grad(): probabilities = torch.softmax(model(ids), dim=1)[0]
    index = probabilities.argmax().item()
    print(f"Prediction: {labels[index]}")
    print(f"Confidence: {probabilities[index].item() * 100:.2f}%")
    print("All probabilities: " + ", ".join(f"{label}={probability:.3f}" for label, probability in zip(labels, probabilities.tolist())))


if __name__ == "__main__":
    main()
