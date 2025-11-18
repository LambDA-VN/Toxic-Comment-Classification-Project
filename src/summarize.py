# summarize_min.py
import torch
from model import HateBERTMultiLabel, MODEL_NAME, NUM_LABELS
from dataset import load_df, k_fold

TRAIN_CSV = "./Dataset/train.csv"
LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]
BATCH_SIZE = 16
EPOCHS = 3
LR_ENC = 2e-5
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.01
MAX_LEN = 256
NUM_WORKERS = 4

def count_params(m):
    t = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return t, tr

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_df(TRAIN_CSV)
    folds = k_fold(df)
    model = HateBERTMultiLabel(model_name=MODEL_NAME, num_labels=NUM_LABELS).to(device)
    total, trainable = count_params(model)

    print("=== Model ===")
    print(f"name={MODEL_NAME}")
    print(f"hidden_size={model.enc.config.hidden_size}")
    print(f"num_labels={NUM_LABELS}")
    print(f"params_total={total}")
    print(f"params_trainable={trainable}")


if __name__ == "__main__":
    main()
