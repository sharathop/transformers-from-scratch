
"""Streamlit interface for the trained full and LoRA AG News classifiers."""

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from transformer_encoder.lora import apply_lora
from transformer_encoder.model import TransformerClassifier


st.set_page_config(
    page_title="AG News Transformer Classifier",
    page_icon="📰",
    layout="wide",
)


def load_vocabulary(path: Path) -> dict[str, int]:
    with path.open("rb") as file:
        vocabulary = pickle.load(file)
    return vocabulary["stoi"]


def encode(text: str, stoi: dict[str, int], max_length: int) -> list[int]:
    unk_id = stoi["<unk>"]
    pad_id = stoi["<pad>"]

    token_ids = [
        stoi.get(token, unk_id)
        for token in text.lower().split()
    ]

    return (token_ids[:max_length] + [pad_id] * max_length)[:max_length]


@st.cache_resource
def load_model(method: str):
    """Load the selected trained model once and cache it."""

    model_dir = Path(os.getenv("MODEL_DIR", "outputs"))
    model_path = model_dir / f"{method}_model.pt"
    vocab_path = model_dir / "vocab.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} was not found. "
            f"Train the model with: python train.py --mode {method} --epochs 5"
        )

    if not vocab_path.exists():
        raise FileNotFoundError(
            f"{vocab_path} was not found. "
            "Run train.py once to create vocab.pkl."
        )

    stoi = load_vocabulary(vocab_path)

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        max_length = checkpoint["max_length"]
        labels = checkpoint["labels"]
        config = checkpoint["model_config"]
        state_dict = checkpoint["state_dict"]
        num_classes = checkpoint["num_classes"]
    else:
        # Compatibility with older raw state-dict checkpoints.
        max_length = 64
        num_classes = 4
        labels = ["World", "Sports", "Business", "Sci/Tech"]
        config = {
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 2,
            "ff_dim": 256,
        }
        state_dict = checkpoint

    model = TransformerClassifier(
        len(stoi),
        max_length=max_length,
        num_classes=num_classes,
        **config,
    )

    if method == "lora":
        apply_lora(model)

    model.load_state_dict(state_dict)
    model.eval()

    return model, stoi, max_length, labels


def classify(text: str, method: str):
    if not text or not text.strip():
        raise ValueError("Enter a news headline or article first.")

    model, stoi, max_length, labels = load_model(method)

    ids = torch.tensor(
        encode(text, stoi, max_length),
        dtype=torch.long,
    ).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(ids), dim=1)[0].tolist()

    scores = dict(zip(labels, probabilities))
    best = max(scores, key=scores.get)

    # Extract attention from the final encoder layer, head 0.
    tokens = text.lower().split()[:max_length]
    attention = (
        model.attention_maps()[-1][0, 0, :len(tokens), :len(tokens)]
        .detach()
        .cpu()
    )

    figure, axis = plt.subplots(
        figsize=(
            max(6, len(tokens) * 0.55),
            max(4, len(tokens) * 0.45),
        )
    )

    image = axis.imshow(attention, cmap="magma", vmin=0, vmax=1)

    axis.set_xticks(range(len(tokens)))
    axis.set_xticklabels(tokens, rotation=45, ha="right")
    axis.set_yticks(range(len(tokens)))
    axis.set_yticklabels(tokens)
    axis.set_xlabel("Key tokens")
    axis.set_ylabel("Query tokens")
    axis.set_title(
        f"{method.title()} attention - final encoder layer, head 0"
    )

    figure.colorbar(image, ax=axis, label="Attention probability")
    figure.tight_layout()

    return best, scores, figure


st.title("📰 AG News Transformer Classifier")

st.markdown(
    """
    Classify a news headline or article using a Transformer Encoder
    built from scratch.

    - **Full fine-tuning:** uses the fully fine-tuned model.
    - **LoRA:** uses the parameter-efficient fine-tuned model.
    """
)

with st.sidebar:
    st.header("Model settings")

    method = st.radio(
        "Choose a model",
        options=["full", "lora"],
        format_func=lambda value: {
            "full": "Full fine-tuning",
            "lora": "LoRA",
        }[value],
    )

    st.caption(
        "The selected model must have its trained checkpoint "
        "available in the outputs directory."
    )

st.subheader("Enter news text")

examples = {
    "Sports headline": (
        "The national team won the championship match"
    ),
    "Business headline": (
        "Central bank raises interest rates to control inflation"
    ),
    "Science and technology headline": (
        "Scientists announce a new telescope mission "
        "to study distant galaxies"
    ),
}

example_choice = st.selectbox(
    "Try an example",
    ["Choose an example..."] + list(examples.keys()),
)

default_text = (
    examples[example_choice]
    if example_choice in examples
    else ""
)

news_text = st.text_area(
    "News headline or article",
    value=default_text,
    height=150,
    placeholder="Paste a headline or article here...",
)

if st.button("Classify news", type="primary"):
    if not news_text.strip():
        st.warning("Enter a news headline or article first.")
    else:
        try:
            with st.spinner("Loading model and classifying news..."):
                prediction, scores, figure = classify(
                    news_text,
                    method,
                )

            st.success(f"Predicted category: {prediction}")

            st.subheader("Class probabilities")

            probability_data = pd.DataFrame(
                {
                    "Category": list(scores.keys()),
                    "Probability": list(scores.values()),
                }
            ).set_index("Category")

            st.bar_chart(probability_data)

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Category": category,
                            "Probability": f"{probability:.2%}",
                        }
                        for category, probability in scores.items()
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("Attention heatmap")
            st.pyplot(figure)
            plt.close(figure)

        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                f"Prediction failed: {error}"
            )
