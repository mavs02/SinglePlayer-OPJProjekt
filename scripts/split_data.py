import pandas as pd
from sklearn.model_selection import train_test_split
import os

# ============================================================
# BASE DIR
# ============================================================
BASE_DIR = "/home/mvlasic/OPJ-SinglePlayer"

DATA_CANDIDATES = [
    f"{BASE_DIR}/data/crofireda_clean.tsv",
    f"{BASE_DIR}/data/raw/crofireda_clean.tsv",
    f"{BASE_DIR}/data_raw/crofireda_clean.tsv"
]

RAW_PATH = None
for path in DATA_CANDIDATES:
    if os.path.exists(path):
        RAW_PATH = path
        break

if RAW_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Expected in data/, data/raw/ or data_raw/"
    )

print("Using dataset:", RAW_PATH)

# ============================================================
# OUTPUT DIR
# ============================================================
OUT_DIR = f"{BASE_DIR}/data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(RAW_PATH, sep="\t", engine="python")
df.columns = df.columns.str.strip().str.lower()

print("Original size:", len(df))

# ============================================================
# CLEAN
# ============================================================
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(int)

df = df[df["text"].str.len() > 0]

X = df["text"].values
y = df["label"].values

# ============================================================
# SPLIT 75 / 5 / 20
# ============================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.8,
    random_state=42,
    stratify=y_temp
)

# ============================================================
# SAVE
# ============================================================
def save(name, X, y):
    path = f"{OUT_DIR}/{name}.csv"
    pd.DataFrame({"text": X, "label": y}).to_csv(path, index=False)
    print(f"Saved {name}: {len(X)} → {path}")

save("train", X_train, y_train)
save("val", X_val, y_val)
save("test", X_test, y_test)

# ============================================================
# SUMMARY
# ============================================================
print("\nDONE SPLIT")
print("Train:", len(X_train))
print("Val:", len(X_val))
print("Test:", len(X_test))