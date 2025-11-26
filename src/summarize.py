import torch
from model import HateBERTMultiLabel, MODEL_NAME, NUM_LABELS

def main():
    device = "cuda"
    model = HateBERTMultiLabel(model_name=MODEL_NAME, num_labels=NUM_LABELS).to(device)
    print(model)

if __name__ == "__main__":
    main()
