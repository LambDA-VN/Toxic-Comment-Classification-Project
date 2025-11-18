import torch.nn as nn
from transformers import AutoModel

NUM_LABELS = 6
MODEL_NAME = "GroNLP/hateBERT"

class HateBERTMultiLabel(nn.Module):
    """HateBERT encoder + linear head → 6 logits (use sigmoid at inference)."""
    def __init__(self, model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS, dropout: float = 0.1):
        super().__init__()
        self.enc = AutoModel.from_pretrained(model_name)
        hidden = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.enc(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        logits = self.head(self.drop(cls))

        return logits
    
