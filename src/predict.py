import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, multilabel_confusion_matrix

from dataset import load_df, Tokenizer, ToxicDataset
from model import HateBERTMultiLabel

TEST_CSV = "./Dataset/test.csv"
MODEL_PATH = "./Models/32b_model_fold1.pt"
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
BATCH_SIZE = 256
MAX_LEN = 512

def infer_probs(df, model, device):
    ds = ToxicDataset(df, Tokenizer(), max_len=MAX_LEN)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    logits_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["labels"].numpy()
            x = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(x["input_ids"], x["attention_mask"]).cpu().numpy()
            logits_all.append(logits)
            labels_all.append(y)
    y_true = np.vstack(labels_all)
    probs = 1.0 / (1.0 + np.exp(-np.vstack(logits_all)))
    return y_true, probs

def tune_thresholds(y_true, probs, grid=None):
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)
    thres = np.full(len(LABELS), 0.5, dtype=np.float32)
    for i in range(len(LABELS)):
        p, y = probs[:, i], y_true[:, i]
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred = (p >= t).astype(int)
            f1 = f1_score(y, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thres[i] = best_t
    return thres

def eval_with_thresholds(y_true, probs, thresholds):
    y_pred = (probs >= thresholds[None, :]).astype(int)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    subset_acc = accuracy_score(y_true, y_pred)
    cm = multilabel_confusion_matrix(y_true, y_pred)

    print("\n=== Thresholds (per label) ===")
    for name, t in zip(LABELS, thresholds):
        print(f"{name:14s}: {t:.2f}")

    print("\n=== Overall ===")
    print(f"Micro F1 : {micro:.4f}")
    print(f"Macro F1 : {macro:.4f}")
    print(f"Subset Accuracy : {subset_acc:.4f}")

    print("\n=== Per-label F1 ===")
    for name, f in zip(LABELS, per_f1):
        print(f"{name:14s}: {f:.4f}")

    print("\n=== Per-label Confusion Matrices (TN FP / FN TP) ===")
    for i, name in enumerate(LABELS):
        tn, fp, fn, tp = cm[i].ravel()
        print(f"{name:14s}: [[{tn} {fp}], [{fn} {tp}]]")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HateBERTMultiLabel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    df = load_df(TEST_CSV)

    y_true, probs = infer_probs(df, model, device)
    thresholds = tune_thresholds(y_true, probs)
    eval_with_thresholds(y_true, probs, thresholds)

if __name__ == "__main__":
    main()
