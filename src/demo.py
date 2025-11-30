import torch
from transformers import AutoTokenizer
from model import HateBERTMultiLabel
import numpy as np
import matplotlib.pyplot as plt

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

MODEL_PATH = r"C:\Users\b2324\Desktop\WorkStation\Sentiment_Analysis\Models\32b_model_fold4.pt"
device = "cuda"

# Auto tokenizer
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hatebert")

# Load model
model = HateBERTMultiLabel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Actual comment from Reddit using API
texts = [
    "Bruh are you even trying? This build is trash. Go read the wiki before posting.",
    "How do you still not understand this? It’s the same question you asked yesterday. Try using your brain.",
    "This update is f**king useless. Devs never listen",
    "Wow, amazing take. Truly the worst comment I’ve read today.",
    "People like you are so *** annoying. Get over yourself."
]

print("Input Texts:")
for t in texts:
    print("-", t)

# Tokenize
enc = tokenizer(
    texts,
    padding="max_length",
    truncation=True,
    max_length=256,
    return_tensors="pt"
).to(device)

# Forward pass
with torch.no_grad():
    logits = model(enc["input_ids"], enc["attention_mask"])         
    probs = torch.sigmoid(logits).cpu().numpy()                     # shape: (batch, labels)

# Print numeric results
print("\nPredicted Probabilities (per comment):")
for idx, (t, p) in enumerate(zip(texts, probs), 1):
    print(f"\nComment {idx}: {t}")
    for label, prob in zip(LABELS, p):
        print(f"  {label}: {prob:.4f}")

