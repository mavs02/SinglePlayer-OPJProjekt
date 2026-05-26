import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report

# ============================================================
# PATHS
# ============================================================

BASE = "/home/mvlasic/OPJ-SinglePlayer"

TRAIN_PATH = f"{BASE}/data/processed/train.csv"
TEST_PATH = f"{BASE}/data/processed/test.csv"
GOLD_PATH = f"{BASE}/data_raw/OPJDataset_clean.csv"

RESULT_PATH = f"{BASE}/results/Trial_1_C_1.0.txt"
PRED_PATH = f"{BASE}/predictions/Prediction_1.csv"

# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
gold_df = pd.read_csv(GOLD_PATH)

# ============================================================
# CLEAN FUNCTION
# ============================================================

def clean(df):
    df = df.copy()

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset mora imati stupce 'text' i 'label'.")

    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df[df["text"] != ""]
    df = df.dropna(subset=["text", "label"])

    df["label"] = df["label"].astype(int)

    return df


train_df = clean(train_df)
test_df = clean(test_df)
gold_df = clean(gold_df)

# ============================================================
# SAFETY CHECKS
# ============================================================

if train_df.empty:
    raise ValueError("Train dataset is empty!")

if test_df.empty:
    raise ValueError("Test dataset is empty!")

if gold_df.empty:
    raise ValueError("Golden OPJ dataset is empty!")

# ============================================================
# FEATURES
# ============================================================

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

X_gold = gold_df["text"]
y_gold = gold_df["label"]

# ============================================================
# MODEL
# ============================================================

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2))),
    ("clf", LogisticRegression(max_iter=300, C=1.0))
])

# ============================================================
# CROSS VALIDATION
# ============================================================

print("\n================ CROSS VALIDATION ================")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]

    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    f1 = f1_score(y_val, preds, average="macro")
    f1_scores.append(f1)

    print(f"Fold {fold}: F1 macro = {f1:.4f}")

cv_mean = np.mean(f1_scores)
cv_std = np.std(f1_scores)

print("\nMean F1 macro:", round(cv_mean, 4))
print("Std:", round(cv_std, 4))

# ============================================================
# FINAL TRAIN
# ============================================================

model.fit(X_train, y_train)

# ============================================================
# TEST EVALUATION - CRO-FIREDA
# ============================================================

pred_test = model.predict(X_test)

test_acc = accuracy_score(y_test, pred_test)
test_f1 = f1_score(y_test, pred_test, average="macro")
test_report = classification_report(y_test, pred_test)

print("\n================ IN-DOMAIN TEST / CRO-FIREDA ================")
print("Accuracy:", round(test_acc, 4))
print("F1 macro:", round(test_f1, 4))
print(test_report)

# ============================================================
# GOLD EVALUATION - OPJ DATASET / NAJDOKTOR
# ============================================================

pred_gold = model.predict(X_gold)

gold_acc = accuracy_score(y_gold, pred_gold)
gold_f1 = f1_score(y_gold, pred_gold, average="macro")
gold_report = classification_report(y_gold, pred_gold)

print("\n================ OUT-OF-DOMAIN TEST / OPJ NAJDOKTOR ================")
print("Accuracy:", round(gold_acc, 4))
print("F1 macro:", round(gold_f1, 4))
print(gold_report)

# ============================================================
# SAVE RESULTS
# ============================================================

with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write("METHOD 1 - ML\n")
    f.write("Model: TF-IDF + Logistic Regression\n")
    f.write("Hyperparameters: C=1.0, max_features=20000, ngram_range=(1,2), max_iter=300\n\n")

    f.write("================ CROSS VALIDATION ================\n")
    for i, score in enumerate(f1_scores, 1):
        f.write(f"Fold {i}: F1 macro = {score:.4f}\n")

    f.write(f"\nMean F1 macro: {cv_mean:.4f}\n")
    f.write(f"Std: {cv_std:.4f}\n\n")

    f.write("================ IN-DOMAIN TEST / CRO-FIREDA ================\n")
    f.write(f"Accuracy: {test_acc:.4f}\n")
    f.write(f"F1 macro: {test_f1:.4f}\n\n")
    f.write(test_report)
    f.write("\n\n")

    f.write("================ OUT-OF-DOMAIN TEST / OPJ NAJDOKTOR ================\n")
    f.write(f"Accuracy: {gold_acc:.4f}\n")
    f.write(f"F1 macro: {gold_f1:.4f}\n\n")
    f.write(gold_report)

print("\nSaved results to:", RESULT_PATH)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "text": X_gold,
    "prediction_label": pred_gold,
    "truth_label": y_gold
})

prediction_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

print("Saved predictions to:", PRED_PATH)