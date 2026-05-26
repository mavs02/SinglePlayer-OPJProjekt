import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset, DataLoader

BASE = "/home/mvlasic/OPJ-SinglePlayer"

TEST_PATH = f"{BASE}/data/processed/test.csv"
GOLD_PATH = f"{BASE}/data_raw/OPJDataset_clean.csv"
MODEL_PATH = f"{BASE}/Method2/DL/cnn_model.pt"

RESULT_PATH = f"{BASE}/results/Trial_3_CNN_improved.txt"
PRED_PATH = f"{BASE}/predictions/Prediction_3.csv"

test_df = pd.read_csv(TEST_PATH)
gold_df = pd.read_csv(GOLD_PATH)

def clean(df):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["text"] != ""]
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df

test_df = clean(test_df)
gold_df = clean(gold_df)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(MODEL_PATH, map_location=device)
vocab = checkpoint["vocab"]

MAX_LEN = 100

def tokenize(text):
    return text.lower().split()

def encode(text):
    tokens = tokenize(text)
    encoded = [vocab.get(token, 1) for token in tokens]
    encoded = encoded[:MAX_LEN]
    while len(encoded) < MAX_LEN:
        encoded.append(0)
    return encoded

class SentimentDataset(Dataset):
    def __init__(self, df):
        self.texts = [encode(text) for text in df["text"]]
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

test_loader = DataLoader(SentimentDataset(test_df), batch_size=32)
gold_loader = DataLoader(SentimentDataset(gold_df), batch_size=32)

class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, 128, kernel_size=3),
            nn.Conv1d(embed_dim, 128, kernel_size=4),
            nn.Conv1d(embed_dim, 128, kernel_size=5)
        ])

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        conv_outputs = []

        for conv in self.convs:
            c = self.relu(conv(x))
            c = self.pool(c).squeeze(-1)
            conv_outputs.append(c)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

model = CNNModel(
    vocab_size=len(vocab),
    embed_dim=200,
    num_classes=3
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])

def evaluate(loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in loader:
            texts = texts.to(device)

            outputs = model(texts)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_labels, all_preds

y_test, pred_test = evaluate(test_loader)

test_acc = accuracy_score(y_test, pred_test)
test_f1 = f1_score(y_test, pred_test, average="macro")
test_report = classification_report(y_test, pred_test)

print("\n================ IMPROVED CNN TEST / CRO-FIREDA ================\n")
print("Accuracy:", round(test_acc, 4))
print("F1 macro:", round(test_f1, 4))
print(test_report)

y_gold, pred_gold = evaluate(gold_loader)

gold_acc = accuracy_score(y_gold, pred_gold)
gold_f1 = f1_score(y_gold, pred_gold, average="macro")
gold_report = classification_report(y_gold, pred_gold)

print("\n================ IMPROVED CNN TEST / OPJ ================\n")
print("Accuracy:", round(gold_acc, 4))
print("F1 macro:", round(gold_f1, 4))
print(gold_report)

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write("METHOD 2 - IMPROVED CNN\n")
    f.write("Model: Embedding + Conv1D kernels 3,4,5 + weighted loss + dropout\n")
    f.write("Hyperparameters: embed_dim=200, filters=128, dropout=0.5, lr=0.0005, epochs=10\n\n")

    f.write("CNN TEST / CRO-FIREDA\n")
    f.write(f"Accuracy: {test_acc:.4f}\n")
    f.write(f"F1 macro: {test_f1:.4f}\n\n")
    f.write(test_report)

    f.write("\n\nCNN TEST / OPJ\n")
    f.write(f"Accuracy: {gold_acc:.4f}\n")
    f.write(f"F1 macro: {gold_f1:.4f}\n\n")
    f.write(gold_report)

prediction_df = pd.DataFrame({
    "text": gold_df["text"],
    "prediction_label": pred_gold,
    "truth_label": y_gold
})

prediction_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

print("\nSaved results:", RESULT_PATH)
print("Saved predictions:", PRED_PATH)