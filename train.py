"""Train and compare full fine-tuning and LoRA, then save attention heatmaps."""
import argparse
import copy
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # File-only backend: works in terminals and headless environments.
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformer_encoder.data import make_splits
from transformer_encoder.lora import apply_lora, trainable_parameters
from transformer_encoder.model import TransformerClassifier


def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for ids, labels, _ in loader:
            logits = model(ids.to(device)); correct += (logits.argmax(1).cpu() == labels).sum().item(); total += len(labels)
    return correct / total


def train(model, train_loader, test_loader, device, epochs, learning_rate):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(); model.to(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for ids, labels, _ in train_loader:
            optimizer.zero_grad(); loss = criterion(model(ids.to(device)), labels.to(device)); loss.backward(); optimizer.step()
        print(f"epoch {epoch + 1:02d}/{epochs}  test_accuracy={evaluate(model, test_loader, device):.3f}")
    elapsed = time.perf_counter() - started
    # CUDA reports actual peak allocation. CPU uses a transparent AdamW-state estimate (2 fp32 moments per trainable weight).
    memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else trainable_parameters(model) * 8 / 2**20
    return {"trainable_parameters": trainable_parameters(model), "training_seconds": round(elapsed, 3), "peak_memory_mb": round(memory, 3), "test_accuracy": round(evaluate(model, test_loader, device), 3)}


def save_attention(model, dataset, device, output_dir):
    model.eval(); ids, _, text = dataset[0]
    with torch.no_grad(): model(ids.unsqueeze(0).to(device))
    tokens = text.lower().split()[:dataset.max_length]
    attention = model.attention_maps()[-1][0, 0, :len(tokens), :len(tokens)].cpu()
    fig, ax = plt.subplots(figsize=(7, 5)); image = ax.imshow(attention, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right"); ax.set_yticks(range(len(tokens)), tokens)
    ax.set_title("Last encoder layer — attention head 0"); fig.colorbar(image, ax=ax, label="attention probability")
    fig.tight_layout(); fig.savefig(output_dir / "attention_heatmap.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare a scratch Transformer trained fully vs. with LoRA.")
    parser.add_argument("--dataset", choices=["ag_news", "toy"], default="ag_news")
    parser.add_argument("--train-samples", type=int, default=30_000, help="AG News: 30,000 by default")
    parser.add_argument("--test-samples", type=int, default=7_600, help="AG News: official 7,600 by default")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5); parser.add_argument("--batch-size", type=int, default=64); parser.add_argument("--seed", type=int, default=7); parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--mode", choices=["both", "full", "lora"], default="both", help="Which training strategy to run")
    args = parser.parse_args(); torch.manual_seed(args.seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    split_args = {"max_length": args.max_length}
    if args.dataset == "ag_news": split_args.update(train_samples=args.train_samples, test_samples=args.test_samples, seed=args.seed)
    train_set, test_set, vocab, num_classes, label_names = make_splits(args.dataset, **split_args)
    print(f"Dataset: {args.dataset}; train={len(train_set):,}, test={len(test_set):,}; labels={label_names}")
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True); test_loader = DataLoader(test_set, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = TransformerClassifier(len(vocab), max_length=train_set.max_length, d_model=128, num_heads=4, num_layers=2, ff_dim=256, num_classes=num_classes)
    all_runs = {"full_finetuning": copy.deepcopy(base), "lora": apply_lora(copy.deepcopy(base))}
    requested = {"both": ["full_finetuning", "lora"], "full": ["full_finetuning"], "lora": ["lora"]}[args.mode]
    runs = {}
    for name in requested:
        model = all_runs[name]
        print(f"\n{name} ({device})")
        runs[name] = train(model, train_loader, test_loader, device, args.epochs, 3e-3)
        checkpoint_name = "full_model.pt" if name == "full_finetuning" else "lora_model.pt"
        torch.save({
            "state_dict": model.state_dict(), "vocab_itos": vocab.itos,
            "max_length": train_set.max_length, "num_classes": num_classes,
            "labels": label_names, "model_config": {"d_model": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 256},
            "method": "full" if name == "full_finetuning" else "lora",
        }, output / checkpoint_name)
        if name == "lora":
            save_attention(model, test_set, device, output)
    with open(output / "comparison.json", "w") as file: json.dump(runs, file, indent=2)
    with open(output / "comparison.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["method", *next(iter(runs.values())).keys()]); writer.writeheader()
        for method, metrics in runs.items(): writer.writerow({"method": method, **metrics})
    print("\n" + json.dumps(runs, indent=2)); print(f"Saved results and attention heatmap to {output}")


if __name__ == "__main__": main()
