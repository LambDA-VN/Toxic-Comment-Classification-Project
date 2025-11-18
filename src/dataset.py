import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer

TEXT_COL = "comment_text" 
LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_NAME = "GroNLP/hateBERT"
MAX_LEN = 256


def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv("./Dataset/train.csv")
    df[LABEL_COLS] = df[LABEL_COLS].astype(int)

    return df

def k_fold(df: pd.DataFrame, n_splits: int = 5, seed: int = 42):
    y = (df[LABEL_COLS].sum(axis=1) > 0).astype(int).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    return list(skf.split(df, y))

def Tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)

class ToxicDataset(Dataset):
    """Create with df.iloc[train_idx] or df.iloc[val_idx]. Tokenizes on access."""
    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int = MAX_LEN):
        self.texts = frame[TEXT_COL].astype(str).tolist()
        self.labels = frame[LABEL_COLS].astype("float32").values
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, i: int):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors ="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()} # input_ids, and attention_mask (Magic stuff happen here)
        item["labels"] = torch.from_numpy(self.labels[i])
        return item


