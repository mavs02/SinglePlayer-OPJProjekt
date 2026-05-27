import os
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


MODEL_DIR = "Method3/Transformer/BERTic/bertic_model"

TEST_SETS = {
    "crofireda_test": "data/processed/test.csv",
    "golden_test": "data_raw/OPJDataset_clean.csv"
}


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(
        f"Column not found. Tried: {possible_names}. Existing columns: {list(df.columns)}"
    )


def load_data(path):
    df = pd.read_csv(path)

    text_col = find_column(df, ["text", "sentence", "review", "content"])
    label_col = find_column(df, ["sentiment", "label", "labels"])

    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def predict(texts, tokenizer, model, device):
    predictions = []
    model.eval()

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(prediction)

    return predictions


def evaluate_dataset(name, path, tokenizer, model, device):
    print(f"\nEvaluating on: {name}")
    print(f"Path: {path}")

    df = load_data(path)

    y_true = df["label"].tolist()
    y_pred = predict(df["text"].tolist(), tokenizer, model, device)

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(y_true, y_pred, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("\nClassification report:")
    print(report)
    print("\nConfusion matrix:")
    print(matrix)

    predictions_df = pd.DataFrame({
        "text": df["text"],
        "true_label": y_true,
        "predicted_label": y_pred,
        "correct": [true == pred for true, pred in zip(y_true, y_pred)]
    })

    predictions_path = f"predictions/Prediction_4_BERTic_{name}.csv"
    results_path = f"results/Trial_4_BERTic_{name}.txt"

    predictions_df.to_csv(predictions_path, index=False)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Method 3: Transformer - BERTic Results on {name}\n")
        f.write("Model: classla/bcms-bertic\n")
        f.write(f"Data: {path}\n\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n\n")
        f.write("Classification report:\n")
        f.write(report)
        f.write("\nConfusion matrix:\n")
        f.write(str(matrix))

    print("Results saved to:", results_path)
    print("Predictions saved to:", predictions_path)


def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("predictions", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading BERTic model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)

    for name, path in TEST_SETS.items():
        evaluate_dataset(name, path, tokenizer, model, device)


if __name__ == "__main__":
    main()