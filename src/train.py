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
def pos_weight_from(labels_array):
    positive_rate = labels_array.mean(axis=0)
    weight_values = (1 - positive_rate) / np.clip(positive_rate, 1e-6, 1.0)
    return torch.tensor(np.clip(weight_values, 1.0, 10.0), dtype=torch.float)


def tune_thresholds(true_labels, predicted_scores):
    thresholds = []
    for idx in range(true_labels.shape[1]):
        scores = predicted_scores[:, idx]
        labels = true_labels[:, idx]

        if labels.sum() == 0:
            thresholds.append(0.5)
            continue

        best_threshold, best_f1 = 0.5, -1.0
        for threshold in np.unique(scores):
            f1 = f1_score(labels, (scores >= threshold).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, threshold

        thresholds.append(float(best_threshold))

    return np.array(thresholds, dtype=np.float32)


def metrics(true_labels, predicted_scores, thresholds):
    predicted_labels = (predicted_scores >= thresholds).astype(int)
    per_label_f1 = [
        f1_score(true_labels[:, i], predicted_labels[:, i], zero_division=0)
        for i in range(true_labels.shape[1])
    ]
    macro_f1 = float(np.mean(per_label_f1))
    micro_f1 = f1_score(true_labels.ravel(), predicted_labels.ravel(), zero_division=0)
    return macro_f1, micro_f1


def run_fold(df, train_index, val_index, tokenizer, device, fold_id):
    train_dataset = ToxicDataset(df.iloc[train_index], tokenizer, max_len=MAX_LENGTH)
    val_dataset = ToxicDataset(df.iloc[val_index], tokenizer, max_len=MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model = HateBERTMultiLabel().to(device)

    train_label_matrix = np.vstack([batch["labels"].numpy() for batch in train_loader])
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_from(train_label_matrix).to(device)
    )

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
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    thresholds = tune_thresholds(labels, probabilities)
    macro_f1, micro_f1 = metrics(labels, probabilities, thresholds)

    save_path = f"./Models/model_fold{fold_id}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved → {save_path}")

    return macro_f1, micro_f1


# ENTRY
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
