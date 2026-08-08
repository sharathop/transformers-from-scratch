# 🚀 Transformer Encoder from Scratch with LoRA

A complete **Transformer Encoder** implementation built **from scratch in PyTorch** without using `torch.nn.Transformer` or `torch.nn.MultiheadAttention`.

This project was built to understand how modern Transformer models work internally by implementing every major component manually, including **Multi-Head Self-Attention**, **Positional Embeddings**, **Feed Forward Networks**, **Residual Connections**, **Layer Normalization**, and **Low-Rank Adaptation (LoRA)**.

The model is trained on the **AG News** dataset for multi-class news classification and compares **Full Fine-Tuning** against **LoRA Fine-Tuning** in terms of accuracy, trainable parameters, memory usage, and training time.

---

# 📌 Project Objective

The goal of this project is to build a Transformer Encoder completely from scratch and understand:

- How token embeddings work
- Why positional embeddings are required
- How Multi-Head Self-Attention computes contextual representations
- How Feed Forward Networks refine token representations
- How Residual Connections and LayerNorm stabilize training
- How LoRA reduces trainable parameters while adapting a model

Unlike typical NLP projects that rely on Hugging Face Transformer implementations, this project manually implements every component using only PyTorch primitives.

---

# ✨ Features

- ✅ Transformer Encoder implemented completely from scratch
- ✅ Multi-Head Self-Attention
- ✅ Learned Token Embeddings
- ✅ Learned Positional Embeddings
- ✅ Feed Forward Network (Linear → GELU → Dropout → Linear)
- ✅ Residual Connections
- ✅ Layer Normalization
- ✅ Padding Mask Support
- ✅ Mean Pooling Sentence Representation
- ✅ LoRA Fine-Tuning
- ✅ Full Fine-Tuning
- ✅ Attention Heatmap Visualization
- ✅ CLI Inference
- ✅ Gradio Web Interface

---

# 🏗️ Architecture

```
                         Input News Article
                                 │
                                 ▼
                          Text Tokenization
                                 │
                                 ▼
                             Token IDs
                                 │
                                 ▼
                          Token Embedding
                                 │
                                 ▼
                      Positional Embedding
                                 │
                                 ▼
                     ┌──────────────────────┐
                     │  Encoder Block × 2   │
                     │                      │
                     │ Multi-Head Attention │
                     │        │             │
                     │ Add & LayerNorm      │
                     │        │             │
                     │ Feed Forward Network │
                     │        │             │
                     │ Add & LayerNorm      │
                     └──────────────────────┘
                                 │
                                 ▼
                        Mean Pooling Layer
                                 │
                                 ▼
                       Linear Classification
                                 │
                                 ▼
                         AG News Prediction
```

---

# 🧠 Transformer Pipeline

The model follows the standard Transformer Encoder pipeline:

```
News Article
      │
      ▼
Tokenization
      │
      ▼
Vocabulary IDs
      │
      ▼
Embedding Layer
      │
      ▼
Positional Embedding
      │
      ▼
Multi-Head Self Attention
      │
      ▼
Residual + LayerNorm
      │
      ▼
Feed Forward Network
      │
      ▼
Residual + LayerNorm
      │
      ▼
Repeat Encoder Blocks
      │
      ▼
Mean Pooling
      │
      ▼
Linear Classifier
      │
      ▼
Prediction
      │
      ▼
Cross Entropy Loss
      │
      ▼
Backpropagation
```

---

# ⚙️ Model Configuration

| Component | Value |
|------------|-------|
| Encoder Blocks | 2 |
| Model Dimension (`d_model`) | 128 |
| Attention Heads | 4 |
| Feed Forward Dimension | 256 |
| Maximum Sequence Length | 64 |
| Number of Classes | 4 |
| Activation | GELU |
| Optimizer | AdamW |
| Loss Function | CrossEntropyLoss |

---

# 🔍 Transformer Components

## 1. Tokenization

The input news article is converted into tokens using a simple whitespace tokenizer.

Example:

```
Apple launches new AI chip
```

↓

```
["apple","launches","new","ai","chip"]
```

---

## 2. Vocabulary IDs

Each token is mapped to an integer ID.

```
apple     → 345
launches  → 912
new       → 105
ai        → 786
chip      → 442
```

These IDs become the model input.

---

## 3. Token Embedding

Each token ID is converted into a dense learnable vector.

Instead of processing integers, the Transformer processes these learned embeddings.

---

## 4. Positional Embedding

Since attention has no notion of word order, learned positional embeddings are added to token embeddings.

This allows the encoder to distinguish between:

```
Dog bites man

vs

Man bites dog
```

---

## 5. Multi-Head Self Attention

Each token is projected into

- Query (Q)
- Key (K)
- Value (V)

Attention is computed using

```
Attention(Q,K,V)

=

softmax(QKᵀ / √dₖ)V
```

This allows every token to gather contextual information from every other token.

Multiple attention heads learn different semantic relationships simultaneously.

---

## 6. Residual Connection + LayerNorm

Residual connections preserve information flow while Layer Normalization stabilizes training.

```
Output = LayerNorm(X + Attention(X))
```

---

## 7. Feed Forward Network

Each contextualized token is independently refined using

```
Linear

↓

GELU

↓

Dropout

↓

Linear
```

Unlike attention, the FFN does **not** communicate across tokens.

It simply learns richer representations for each token.

---

## 8. Mean Pooling

The final encoder outputs are averaged (excluding padding tokens) to produce a single sentence representation.

---

## 9. Classification Head

The pooled sentence embedding is passed through a linear classifier to predict one of four AG News categories:

- World
- Sports
- Business
- Sci/Tech

---

# 📚 Dataset

This project uses the **AG News** dataset from Hugging Face.

### Original Dataset

- Training Samples: **120,000**
- Test Samples: **7,600**
- Classes: **4**

This implementation trains on

- **30,000** randomly shuffled training samples
- **7,600** official test samples

---

# 🔥 LoRA Fine-Tuning

Instead of updating every parameter, LoRA freezes the original model weights and learns a low-rank update.

The adapted projection becomes

```
W' = W + (B × A) × α / r
```

where

- W = frozen weight
- A = down projection
- B = up projection
- r = rank

In this project,

LoRA adapters are inserted into

- Query Projection (WQ)
- Value Projection (WV)

inside every encoder block.

Only

- LoRA adapters
- Classification Head

remain trainable.

---

# 📊 Results

| Method | Epochs | Accuracy | Trainable Parameters | Training Time | Memory |
|----------|---------|------------|---------------------|----------------|------------|
| Full Fine-Tuning | 5 | **87.5%** | 4,114,180 | 10.4 min | 31.389 MB |
| LoRA | 15 | **72.6%** | 4,612 | 23.0 min | 0.035 MB |

### Observation

Although LoRA trains **~892× fewer parameters**, it achieves lower accuracy because this project trains the Transformer **from scratch**.

LoRA is designed primarily for adapting **pretrained** models rather than training randomly initialized networks.

---

# 🎯 Attention Visualization

The project visualizes the final encoder attention weights.

The generated heatmap shows

- Query Tokens (Rows)
- Key Tokens (Columns)

Brighter regions indicate stronger attention scores.

This provides insight into which words influence one another during prediction.

---

# 💻 Train the Model

Train both Full Fine-Tuning and LoRA

```bash
python train.py --mode both --epochs 5
```

Train only Full Fine-Tuning

```bash
python train.py --mode full --epochs 5
```

Train only LoRA

```bash
python train.py --mode lora --epochs 15
```

---

# 🖥️ Predict Custom News

Using the Full Fine-Tuned model

```bash
python predict.py --method full --text "Apple launches new AI processor"
```

Using LoRA

```bash
python predict.py --method lora --text "The football team won the championship"
```

---

# 🌐 Launch Web Interface

```bash
python app.py
```

A local Gradio application will open for interactive news classification.

---

# 📁 Project Structure

```
Transformer-from-Scratch/
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
│
├── transformer_encoder/
│   ├── data.py
│   ├── model.py
│   └── lora.py
│
└── outputs/
    ├── comparison.csv
    ├── comparison.json
    ├── full_model.pt
    ├── lora_model.pt
    └── attention_heatmap.png
```

---

# 🛠️ Tech Stack

- Python
- PyTorch
- Hugging Face Datasets
- Matplotlib
- Gradio

---

# 📖 Key Learnings

Through this project I gained a practical understanding of:

- Transformer Encoder architecture
- Multi-Head Self-Attention
- Positional Embeddings
- Feed Forward Networks
- Residual Connections
- Layer Normalization
- Backpropagation through Transformers
- LoRA (Low-Rank Adaptation)
- Parameter-Efficient Fine-Tuning
- Attention Visualization
- End-to-End NLP Model Development

---

# ⭐ Future Improvements

- Replace the whitespace tokenizer with Byte Pair Encoding (BPE)
- Train on the complete AG News dataset
- Add mixed-precision (FP16) training
- Support pretrained Transformer initialization
- Implement Decoder-only GPT architecture
- Implement Encoder-Decoder Transformer for machine translation
- Add TensorBoard/W&B experiment tracking

---

## 👨‍💻 Author

**Sharath M**

Built to deeply understand the internal mechanics of Transformer architectures and parameter-efficient fine-tuning techniques rather than relying on high-level APIs.