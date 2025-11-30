import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score

from dataset import load_df, Tokenizer, ToxicDataset
from model import HateBERTMultiLabel

TEST_CSV = "./Dataset/test.csv"
MODEL_PATH = "./Models/32b_model_fold4.pt"
BATCH_SIZE = 32
MAX_LEN = 256
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def main():
    device = "cuda"

    df = load_df(TEST_CSV)
    ds = ToxicDataset(df, Tokenizer(), max_len=MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model = HateBERTMultiLabel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    logits_list, labels_list = [], []

    with torch.no_grad():
        for batch in loader:
            labels_list.append(batch["labels"].numpy())
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"]).cpu().numpy()
            logits_list.append(logits)

    logits = np.vstack(logits_list)
    labels = np.vstack(labels_list)
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
    p_macro = precision_score(labels, preds, average="macro", zero_division=0)
    p_micro = precision_score(labels, preds, average="micro", zero_division=0)

    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 Macro     : {f1_macro:.4f}")
    print(f"F1 Micro     : {f1_micro:.4f}")
    print(f"Precision Mac: {p_macro:.4f}")
    print(f"Precision Mic: {p_micro:.4f}")

if __name__ == "__main__":
    main()
