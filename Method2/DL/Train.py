import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from collections import Counter

BASE = "/home/mvlasic/OPJ-SinglePlayer"

TRAIN_PATH = f"{BASE}/data/processed/train.csv"
VAL_PATH = f"{BASE}/data/processed/val.csv"

MODEL_PATH = f"{BASE}/Method2/DL/cnn_model.pt"

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

def clean(df):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["text"] != ""]
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df

train_df = clean(train_df)
val_df = clean(val_df)

def tokenize(text):
    return text.lower().split()

counter = Counter()

for text in train_df["text"]:
    counter.update(tokenize(text))

vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(30000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

MAX_LEN = 100

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

train_loader = DataLoader(SentimentDataset(train_df), batch_size=32, shuffle=True)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNNModel(
    vocab_size=len(vocab),
    embed_dim=200,
    num_classes=3
).to(device)

class_counts = train_df["label"].value_counts().sort_index()
total = class_counts.sum()
weights = total / (len(class_counts) * class_counts)
weights = torch.tensor(weights.values, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

EPOCHS = 38

print("\n================ TRAINING IMPROVED CNN ================\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for texts, labels in train_loader:
        texts = texts.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(texts)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss:.4f}")

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab": vocab
}, MODEL_PATH)

print("\nSaved improved CNN model:", MODEL_PATH)