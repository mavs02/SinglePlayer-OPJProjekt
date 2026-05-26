import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ============================================================
# PATH
# ============================================================
DATA_PATH = "/home/mvlasic/OPJ-SinglePlayer/data_raw/crofireda_train.tsv"
RESULTS_PATH = "/home/mvlasic/OPJ-SinglePlayer/results/training_report.txt"

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH, sep="\t", engine="python")
df.columns = df.columns.str.strip().str.lower()

print("Dataset size:", len(df))

# ============================================================
# CLEAN TEXT (minimalno, bez gubitka podataka)
# ============================================================
def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

df["text"] = df["text"].apply(clean_text)

# ============================================================
# LABELS
# ============================================================
df["label"] = df["label"].astype(int)

# ============================================================
# FEATURES
# ============================================================
X = df["text"].values
y = df["label"].values

# safety check
if len(X) == 0:
    raise ValueError("Dataset prazan!")

# ============================================================
# MODEL
# ============================================================
model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000)),
    ("clf", LogisticRegression(max_iter=300))
])

# ============================================================
# TRAIN
# ============================================================
model.fit(X, y)

# ============================================================
# EVALUATION (train-only, jednostavno)
# ============================================================
pred = model.predict(X)

acc = accuracy_score(y, pred)
f1 = f1_score(y, pred, average="macro")
report = classification_report(y, pred)

print("Accuracy:", acc)
print("F1 macro:", f1)

# ============================================================
# SAVE RESULTS (bez modela)
# ============================================================
with open(RESULTS_PATH, "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"F1 macro: {f1}\n\n")
    f.write(report)

print("DONE - results saved")