import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

BASE = "/home/mvlasic/OPJ-SinglePlayer"

TRAIN_PATH = f"{BASE}/data/processed/train.csv"
TEST_PATH = f"{BASE}/data/processed/test.csv"
GOLD_PATH = f"{BASE}/data_raw/OPJDataset_clean.csv"

RESULT_PATH = f"{BASE}/results/Trial_3_MLP_balanced.txt"
PRED_PATH = f"{BASE}/predictions/Prediction_3.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
gold_df = pd.read_csv(GOLD_PATH)

def clean(df):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["text"] != ""]
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    return df

train_df = clean(train_df)
test_df = clean(test_df)
gold_df = clean(gold_df)

X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

X_gold = gold_df["text"]
y_gold = gold_df["label"]

def build_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True
        )),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        ))
    ])

print("\n================ METHOD 2 / DL - CROSS VALIDATION ================")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train), 1):
    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]

    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    model = build_model()

    weights = compute_sample_weight(class_weight="balanced", y=y_tr)

    model.fit(X_tr, y_tr, clf__sample_weight=weights)

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

final_model = build_model()

final_weights = compute_sample_weight(class_weight="balanced", y=y_train)

final_model.fit(X_train, y_train, clf__sample_weight=final_weights)

# ============================================================
# IN-DOMAIN TEST
# ============================================================

pred_test = final_model.predict(X_test)

test_acc = accuracy_score(y_test, pred_test)
test_f1 = f1_score(y_test, pred_test, average="macro")
test_report = classification_report(y_test, pred_test)

print("\n================ IN-DOMAIN TEST / CRO-FIREDA ================")
print("Accuracy:", round(test_acc, 4))
print("F1 macro:", round(test_f1, 4))
print(test_report)

# ============================================================
# OUT-OF-DOMAIN TEST / OPJ
# ============================================================

pred_gold = final_model.predict(X_gold)

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
    f.write("METHOD 2 - DL BASELINE\n")
    f.write("Model: TF-IDF + Balanced MLPClassifier\n")
    f.write("Hyperparameters: hidden_layer_sizes=(128,64), alpha=0.001, max_iter=200, early_stopping=True, class_weight=balanced via sample_weight\n\n")

    f.write("================ CROSS VALIDATION ================\n")
    for i, score in enumerate(f1_scores, 1):
        f.write(f"Fold {i}: F1 macro = {score:.4f}\n")

    f.write(f"\nMean F1 macro: {cv_mean:.4f}\n")
    f.write(f"Std: {cv_std:.4f}\n\n")

    f.write("================ IN-DOMAIN TEST / CRO-FIREDA ================\n")
    f.write(f"Accuracy: {test_acc:.4f}\n")
    f.write(f"F1 macro: {test_f1:.4f}\n\n")
    f.write(test_report)

    f.write("\n\n================ OUT-OF-DOMAIN TEST / OPJ NAJDOKTOR ================\n")
    f.write(f"Accuracy: {gold_acc:.4f}\n")
    f.write(f"F1 macro: {gold_f1:.4f}\n\n")
    f.write(gold_report)

prediction_df = pd.DataFrame({
    "text": X_gold,
    "prediction_label": pred_gold,
    "truth_label": y_gold
})

prediction_df.to_csv(PRED_PATH, index=False, encoding="utf-8-sig")

print("\nSaved results to:", RESULT_PATH)
print("Saved predictions to:", PRED_PATH)