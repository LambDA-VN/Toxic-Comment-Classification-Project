import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dataset import load_df, k_fold, Tokenizer, ToxicDataset
from model import HateBERTMultiLabel

# HYPERPARAMETERS
TRAIN_CSV = "./Dataset/train.csv"
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE_ENCODER = 2e-5
LEARNING_RATE_HEAD = 1e-4
WEIGHT_DECAY = 0.01
MAX_LENGTH = 256
NUM_WORKERS = 4

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


# HELPERS
def pos_weight_from(y):
    import numpy as np
    # same formula, just using numpy fully (library)
    positive = y.mean(axis=0)
    weights = (1 - positive) / np.clip(positive, 1e-6, 1.0)
    return torch.tensor(np.clip(weights, 1.0, 10.0), dtype=torch.float)


def tune_thresholds(y_true, y_score):
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
    from sklearn.metrics import f1_score
    y_pred = (y_score >= thr).astype(int)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return float(macro), float(micro)


def run_fold(df, train_index, val_index, tokenizer, device, fold_id):
    train_dataset = ToxicDataset(df.iloc[train_index], tokenizer, max_len=MAX_LENGTH)
    val_dataset = ToxicDataset(df.iloc[val_index], tokenizer, max_len=MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = HateBERTMultiLabel().to(device)

    train_label_matrix = np.vstack([batch["labels"].numpy() for batch in train_loader])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_from(train_label_matrix).to(device))

    optimizer = torch.optim.AdamW(
        [
            {"params": model.enc.parameters(), "lr": LEARNING_RATE_ENCODER},
            {"params": model.head.parameters(), "lr": LEARNING_RATE_HEAD},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # TRAINING LOOP
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn(logits, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 100 == 0 or step == len(train_loader):
                avg = total_loss / step
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(train_loader)} | Loss {avg:.4f}", flush=True)

        print(f"Epoch {epoch+1} done | Avg loss {total_loss/len(train_loader):.4f}", flush=True)

    # VALIDATION
    model.eval()
    logits_list, label_list = [], []

    with torch.no_grad():
        for batch in val_loader:
            labels = batch["labels"].numpy()
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"]).cpu().numpy()

            logits_list.append(logits)
            label_list.append(labels)

    logits = np.vstack(logits_list)
    labels = np.vstack(label_list)

    probabilities = torch.sigmoid(torch.from_numpy(logits)).numpy()

    thresholds = tune_thresholds(labels, probabilities)
    macro_f1, micro_f1 = metrics(labels, probabilities, thresholds)

    save_path = f"./Model/model_fold{fold_id}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")

    return macro_f1, micro_f1

# ENTRY
def main():
    device = "cuda"
    df = load_df(TRAIN_CSV)
    tokenizer = Tokenizer()
    folds = k_fold(df)

    macro_scores, micro_scores = [], []

    for fold_id, (train_index, val_index) in enumerate(folds, 1):
        macro, micro = run_fold(df, train_index, val_index, tokenizer, device, fold_id)
        print(f"Fold {fold_id}: macro_f1={macro:.4f}  micro_f1={micro:.4f}")
        macro_scores.append(macro)
        micro_scores.append(micro)

    print(
        f"Avg over {len(folds)} folds → macro_f1={np.mean(macro_scores):.4f}  micro_f1={np.mean(micro_scores):.4f}"
    )


if __name__ == "__main__":
    main()
