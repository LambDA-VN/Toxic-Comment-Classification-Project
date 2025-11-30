# Hate Speech / Toxic Comment Multi-Label Classifier (HateBERT)

This project fine-tunes **HateBERT (GroNLP/hateBERT)** using PyTorch for **multi-label toxic comment classification** across 6 labels.
Model link: https://drive.google.com/file/d/16WNrfyjBX7PaqP2yP64aN4r3qPYu9d_x/view?usp=sharing

## Labels

```
toxic
severe_toxic
obscene
threat
insult
identity_hate
```

## Model Architecture

* Pretrained **HateBERT encoder**
* Dropout layer
* Linear classification head → 6 logits
* **Sigmoid activation at inference** for independent label probabilities

### Output Format

| Stage     | Output                           |
| --------- | -------------------------------- |
| Training  | Raw logits + `BCEWithLogitsLoss` |
| Inference | Sigmoid probabilities per label  |

## Training Strategy

* **5-Fold Stratified K-Fold cross-validation**
* Class imbalance handled via **positive class weighting**
* Per-label classification thresholds tuned using **best F1 score per class**
* Trained models saved per fold to:

```
./Model/model_foldX.pt
```

## Project Structure

```
📦 project
├── dataset.py     → data loading + tokenizer + k-fold split + PyTorch dataset
├── model.py       → HateBERT encoder + linear multi-label head
├── train.py       → training loop, validation, threshold tuning, and metrics
├── Dataset/
│   └── train.csv  → toxic comment dataset (must contain 6 label columns)
└── Model/
    └── model_foldX.pt  → trained weights per fold
```

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd project
```

### 2. Install dependencies

Create `requirements.txt` with:

```
torch
numpy
pandas
scikit-learn
scikit-learn
transformers
sentencepiece
```

Then run:

```bash
pip install -r requirements.txt
```

## Usage

### 🧠 Train with 5-fold cross-validation

```bash
python train.py
```

### ✔ Training Output Sample

```
Epoch 1/5 | Step 100/xxx | Loss 0.2134
Epoch 1 done | Avg loss 0.1987
...
Fold 3: macro_f1=0.8124  micro_f1=0.8843
Avg over 5 folds → macro_f1=0.7998  micro_f1=0.8721
```

### 🔍 Run Inference on New Text

Example script:

```python
import torch
import numpy as np
from transformers import AutoTokenizer
from model import HateBERTMultiLabel, MODEL_NAME, NUM_LABELS

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = HateBERTMultiLabel().to(device)
model.load_state_dict(torch.load("./Model/model_fold1.pt"))
model.eval()

def predict(text, thresholds=None):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    
    with torch.no_grad():
        logits = model(enc["input_ids"], enc["attention_mask"])
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    if thresholds is None:
        return probs  # return raw probabilities
    return (probs >= thresholds).astype(int)

# Example
sample = "You are an idiot and a danger to everyone."
probs = predict(sample)
print("Probabilities:", probs)
```

## Notes

* `Dataset/train.csv` must include column **comment_text** and the 6 label columns
* Loss uses **logits directly (no sigmoid during training)**
* Threshold tuning selects the best F1 threshold per label **independently**

## Future Improvements

Consider adding:

* Mixed precision training (FP16/BF16)
* Learning rate scheduling
* Model ensembling across folds
* Inference wrapper API (FastAPI/Flask)


