"""Gradio interface for testing the trained full and LoRA AG News classifiers."""
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from transformer_encoder.data import Vocabulary, make_splits
from transformer_encoder.lora import apply_lora
from transformer_encoder.model import TransformerClassifier


def vocabulary_from_tokens(tokens):
    vocab = Vocabulary([], min_frequency=1)
    vocab.itos = tokens
    vocab.stoi = {word: index for index, word in enumerate(tokens)}
    return vocab


@lru_cache(maxsize=2)
def load_model(method: str):
    """Load a saved checkpoint once and keep it in memory for later UI requests."""
    path = Path("outputs") / f"{method}_model.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} was not found. Train with: python train.py --mode {method} --epochs 5")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        vocab = vocabulary_from_tokens(checkpoint["vocab_itos"])
        max_length, labels = checkpoint["max_length"], checkpoint["labels"]
        config, state_dict = checkpoint["model_config"], checkpoint["state_dict"]
        num_classes = checkpoint["num_classes"]
    else:
        # Supports LoRA files saved before the metadata-aware checkpoint update.
        _, _, vocab, num_classes, labels = make_splits("ag_news", train_samples=30_000, test_samples=7_600, max_length=64, seed=7)
        max_length, config, state_dict = 64, {"d_model": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 256}, checkpoint
    model = TransformerClassifier(len(vocab), max_length=max_length, num_classes=num_classes, **config)
    if method == "lora":
        apply_lora(model)
    model.load_state_dict(state_dict)
    model.eval()
    return model, vocab, max_length, labels


def classify(text: str, method: str):
    if not text or not text.strip():
        raise gr.Error("Enter a news headline or article first.")
    model, vocab, max_length, labels = load_model(method)
    ids = torch.tensor(vocab.encode(text, max_length)).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(ids), dim=1)[0].tolist()
    scores = dict(zip(labels, probabilities))
    best = max(scores, key=scores.get)
    tokens = text.lower().split()[:max_length]
    attention = model.attention_maps()[-1][0, 0, :len(tokens), :len(tokens)].cpu()
    figure, axis = plt.subplots(figsize=(max(6, len(tokens) * 0.55), max(4, len(tokens) * 0.45)))
    image = axis.imshow(attention, cmap="magma", vmin=0, vmax=1)
    axis.set_xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    axis.set_yticks(range(len(tokens)), tokens)
    axis.set_xlabel("Key tokens")
    axis.set_ylabel("Query tokens")
    axis.set_title(f"{method.title()} attention - final encoder layer, head 0")
    figure.colorbar(image, ax=axis, label="Attention probability")
    figure.tight_layout()
    return f"{best} ({scores[best] * 100:.2f}% confidence)", scores, figure


with gr.Blocks(title="AG News Transformer Classifier") as demo:
    gr.Markdown("# AG News Transformer Classifier\nClassify a news headline or article with the scratch Transformer. Choose **Full fine-tuning** for the higher-accuracy model or **LoRA** for the parameter-efficient model.")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="News text", lines=6, placeholder="Paste a headline or article here...")
            method = gr.Radio([("Full fine-tuning", "full"), ("LoRA", "lora")], value="full", label="Model")
            button = gr.Button("Classify", variant="primary")
        with gr.Column():
            prediction = gr.Textbox(label="Prediction")
            probabilities = gr.Label(label="Class probabilities", num_top_classes=4)
    attention_plot = gr.Plot(label="Attention heatmap")
    gr.Examples(
        examples=[
            ["The national team won the championship match", "full"],
            ["Central bank raises interest rates to control inflation", "full"],
            ["Scientists announce a new telescope mission to study distant galaxies", "full"],
        ], inputs=[text, method], label="Examples",
    )
    button.click(classify, inputs=[text, method], outputs=[prediction, probabilities, attention_plot])


if __name__ == "__main__":
    demo.launch()
