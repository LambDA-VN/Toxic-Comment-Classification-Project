import os
import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

# Use non-GUI backend to prevent tkinter crashes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  

from dataset import load_df, k_fold, Tokenizer, ToxicDataset
from model import HateBERTMultiLabel

TRAIN_CSV = "./Dataset/train.csv"
BATCH_SIZE = 32
EPOCHS = 3
LR_ENC = 2e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.01
MAX_LEN = 256
NUM_WORKERS = 4

LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]

def pos_weight_from(y):
    """Compute positive class weights for BCE loss."""
    p = y.mean(axis=0)
    w = (1 - p) / np.clip(p, 1e-6, 1.0)
    return torch.tensor(np.clip(w, 1.0, 10.0), dtype=torch.float)

def tune_thresholds(y_true, y_score):
    """Tune thresholds per class to maximize F1."""
    thr = []
    for i in range(y_true.shape[1]):
        s, y = y_score[:, i], y_true[:, i]
        if y.sum() == 0:
            thr.append(0.5)
            continue
        best_t, best_f1 = 0.5, -1.0
        for t in np.unique(s):
            f1 = f1_score(y, (s >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thr.append(float(best_t))
    return np.array(thr, dtype=np.float32)

def metrics(y_true, y_score, thr):
    """Compute macro and micro F1 given tuned thresholds."""
    y_pred = (y_score >= thr).astype(int)
    per_f1 = [f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
              for i in range(y_true.shape[1])]
    macro = float(np.mean(per_f1))
    micro = f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)
    return macro, micro

def run_fold(df, tr_idx, va_idx, tok, device, fold_id):
    train_ds = ToxicDataset(df.iloc[tr_idx], tok, max_len=MAX_LEN)
    val_ds   = ToxicDataset(df.iloc[va_idx], tok, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = HateBERTMultiLabel().to(device)

    # compute positive weights for class imbalance
    y_train = np.vstack([b["labels"].numpy() for b in train_loader])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(y_train).to(device))

    opt = torch.optim.AdamW(
        [{"params": model.enc.parameters(), "lr": LR_ENC},
         {"params": model.head.parameters(), "lr": LR_HEAD}],
        weight_decay=WEIGHT_DECAY,
    )

    # metric trackers
    train_losses, val_f1s, val_accs = [], [], []
    step_losses, step_counts = [], []  # for smooth fractional plot

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, batch["labels"])
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()

            # record every 100 steps for fractional epoch
            if step % 100 == 0:
                step_losses.append(loss.item())
                step_counts.append(epoch + step / len(train_loader))
                avg = total_loss / step
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss {avg:.4f}", flush=True)

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1} done | Avg loss {avg_loss:.4f}", flush=True)

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch["labels"].numpy()
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(batch["input_ids"], batch["attention_mask"]).cpu().numpy()
                all_logits.append(logits); all_labels.append(labels)

        logits = np.vstack(all_logits)
        labels = np.vstack(all_labels)
        probs  = 1.0 / (1.0 + np.exp(-logits))
        preds  = (probs >= 0.5).astype(int)

        f1_val = f1_score(labels.ravel(), preds.ravel(), average="macro", zero_division=0)
        acc_val = (preds == labels).mean()
        val_f1s.append(f1_val)
        val_accs.append(acc_val)
        print(f"Validation after Epoch {epoch+1}: F1={f1_val:.4f}, Accuracy={acc_val:.4f}")

    thr = tune_thresholds(labels, probs)
    macro, micro = metrics(labels, probs, thr)

    os.makedirs("./Models", exist_ok=True)
    save_path = f"./Models/32b_model_fold{fold_id}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")

    np.savez(f"./Models/metrics_fold{fold_id}.npz",
             loss=np.array(train_losses),
             f1=np.array(val_f1s),
             acc=np.array(val_accs))
    print(f"Metrics saved → ./Models/metrics_fold{fold_id}.npz")

    plt.figure(figsize=(7,4))
    plt.plot(step_counts, step_losses, alpha=0.7, label="Batch Loss (smooth)")
    plt.plot(range(1, len(val_f1s)+1), val_f1s, marker='s', label="Validation F1")
    plt.plot(range(1, len(val_accs)+1), val_accs, marker='^', label="Validation Accuracy")
    plt.xlabel("Epoch (fractional)")
    plt.ylabel("Score / Loss")
    plt.title(f"Fold {fold_id} Training Progress")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"./Models/fold_progress_{fold_id}.png")
    plt.close()

    return macro, micro


def main():
    device = "cuda"
    df = load_df(TRAIN_CSV)
    tok = Tokenizer()
    folds = k_fold(df)

    macros, micros = [], []
    for k, (tr_idx, va_idx) in enumerate(folds, 1):
        macro, micro = run_fold(df, tr_idx, va_idx, tok, device, fold_id=k)
        print(f"Fold {k}: macro_f1={macro:.4f}  micro_f1={micro:.4f}")
        macros.append(macro); micros.append(micro)

    print(f"Avg over {len(folds)} folds → macro_f1={np.mean(macros):.4f}  micro_f1={np.mean(micros):.4f}")

if __name__ == "__main__":
    main()
