import re
import os
import pandas as pd
import torch

from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

BASE = "/home/mvlasic/OPJ-SinglePlayer"

MODEL_NAME = "google/gemma-3-4b-it"

DATASETS = [
    {
        "name": "CROFIREDA_TEST",
        "display_name": "Test 1: Cro-FiReDa",
        "path": f"{BASE}/data/processed/test.csv",
        "result_path": f"{BASE}/results/Trial_4_Gemma_zero_shot_crofireda.txt",
        "prediction_path": f"{BASE}/predictions/Prediction_Gemma_zero_shot_crofireda.csv"
    },
    {
        "name": "GOLDEN_DATASET",
        "display_name": "Test 2: Golden dataset",
        "path": f"{BASE}/data_raw/OPJDataset_clean_review_labels_final.csv",
        "result_path": f"{BASE}/results/Trial_4_Gemma_zero_shot_golden.txt",
        "prediction_path": f"{BASE}/predictions/Prediction_Gemma_zero_shot_golden.csv"
    }
]


def find_label_column(df):
    possible_columns = [
        "label",
        "label_m",
        "final_label",
        "sentiment",
        "truth_label",
        "true_label"
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(f"No label column found. Available columns: {list(df.columns)}")


def clean(df):
    df = df.copy()

    if "text" not in df.columns:
        raise ValueError(f"No text column found. Available columns: {list(df.columns)}")

    label_col = find_label_column(df)

    df["text"] = df["text"].astype(str).str.strip()
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")

    df = df[df["text"] != ""]
    df = df.dropna(subset=["text", label_col])
    df[label_col] = df[label_col].astype(int)

    df = df.rename(columns={label_col: "label"})

    return df[["text", "label"]]


def build_prompt(text):
    return f"""
You are a sentiment classification model for Croatian reviews.

Classify the following Croatian sentence into exactly one sentiment label.

Labels:
0 = negative
1 = neutral
2 = positive

Return only one number: 0, 1, or 2.

Sentence:
{text}

Answer:
"""


def extract_label(output_text):
    match = re.search(r"\b[012]\b", output_text)
    if match:
        return int(match.group(0))
    return -1


def evaluate_dataset(generator, dataset_config):
    dataset_name = dataset_config["display_name"]
    dataset_path = dataset_config["path"]
    result_path = dataset_config["result_path"]
    prediction_path = dataset_config["prediction_path"]

    print(f"\n================ GEMMA ZERO-SHOT / {dataset_name} ================\n")

    df = pd.read_csv(dataset_path)
    df = clean(df)

    X = df["text"]
    y = df["label"]

    print("Dataset:", dataset_path)
    print("Number of examples:", len(df))
    print("Model:", MODEL_NAME)

    predictions = []
    raw_outputs = []

    print("\nRunning Gemma zero-shot sentiment classification...\n")

    for i, text in enumerate(X.tolist(), start=1):
        prompt = build_prompt(text)

        result = generator(
            prompt,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id
        )

        generated_text = result[0]["generated_text"]
        answer = generated_text.replace(prompt, "").strip()

        prediction = extract_label(answer)

        predictions.append(prediction)
        raw_outputs.append(answer)

        if i % 50 == 0:
            print(f"Processed {i}/{len(X)} examples")

    prediction_df = pd.DataFrame({
        "text": X,
        "truth_label": y,
        "prediction_label": predictions,
        "raw_output": raw_outputs
    })

    prediction_df["correct"] = prediction_df["truth_label"] == prediction_df["prediction_label"]

    prediction_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    valid_df = prediction_df[prediction_df["prediction_label"].isin([0, 1, 2])]

    if len(valid_df) == 0:
        raise ValueError(f"Gemma did not return any valid labels for {dataset_name}.")

    y_true = valid_df["truth_label"]
    y_pred = valid_df["prediction_label"]

    accuracy = accuracy_score(y_true, y_pred)

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["negative", "neutral", "positive"],
        zero_division=0
    )

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"\n================ TEST RESULTS / {dataset_name} ================\n")
    print("Accuracy:", round(accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))
    print(report)
    print("\nConfusion matrix:")
    print(matrix)

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"METHOD 4 - GENAI / GEMMA ZERO-SHOT - {dataset_name}\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write("Method: Zero-shot prompt-based sentiment classification\n")
        f.write("Label scheme: 0 = negative, 1 = neutral, 2 = positive\n\n")

        f.write("================ TEST RESULTS ================\n")
        f.write(f"Number of examples: {len(prediction_df)}\n")
        f.write(f"Valid predictions: {len(valid_df)}\n\n")

        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Weighted precision: {weighted_precision:.4f}\n")
        f.write(f"Weighted recall: {weighted_recall:.4f}\n")
        f.write(f"Weighted F1-score: {weighted_f1:.4f}\n\n")

        f.write(f"Macro precision: {macro_precision:.4f}\n")
        f.write(f"Macro recall: {macro_recall:.4f}\n")
        f.write(f"Macro F1-score: {macro_f1:.4f}\n\n")

        f.write("Classification report:\n")
        f.write(report)

        f.write("\nConfusion matrix:\n")
        f.write(str(matrix))

        f.write("\n\nIncorrect predictions, first 20:\n")
        wrong = prediction_df[prediction_df["correct"] == False].head(20)

        if len(wrong) == 0:
            f.write("No incorrect predictions.\n")
        else:
            for _, row in wrong.iterrows():
                f.write(f"\nText: {row['text']}\n")
                f.write(f"Truth label: {row['truth_label']}\n")
                f.write(f"Prediction label: {row['prediction_label']}\n")
                f.write(f"Raw output: {row['raw_output']}\n")

    print("\nSaved results to:", result_path)
    print("Saved predictions to:", prediction_path)

    return {
        "dataset": dataset_name,
        "precision": weighted_precision,
        "recall": weighted_recall,
        "f1": weighted_f1,
        "accuracy": accuracy,
        "macro_f1": macro_f1
    }


def main():
    print("\n================ METHOD 4 / GENAI - GEMMA ZERO-SHOT ================\n")

    os.makedirs(f"{BASE}/results", exist_ok=True)
    os.makedirs(f"{BASE}/predictions", exist_ok=True)

    print("Loading Gemma model...")
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    summary_results = []

    for dataset_config in DATASETS:
        result = evaluate_dataset(generator, dataset_config)
        summary_results.append(result)

    summary_path = f"{BASE}/results/Trial_4_Gemma_zero_shot_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("METHOD 4 - GENAI / GEMMA ZERO-SHOT SUMMARY\n")
        f.write(f"Model: {MODEL_NAME}\n\n")
        f.write("| Dataset | Precision | Recall | F1 | Accuracy | Macro F1 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")

        for item in summary_results:
            f.write(
                f"| {item['dataset']} | "
                f"{item['precision']:.4f} | "
                f"{item['recall']:.4f} | "
                f"{item['f1']:.4f} | "
                f"{item['accuracy']:.4f} | "
                f"{item['macro_f1']:.4f} |\n"
            )

    print("\n================ GEMMA SUMMARY ================\n")
    for item in summary_results:
        print(
            f"{item['dataset']} | "
            f"Precision: {item['precision']:.4f} | "
            f"Recall: {item['recall']:.4f} | "
            f"F1: {item['f1']:.4f} | "
            f"Accuracy: {item['accuracy']:.4f} | "
            f"Macro F1: {item['macro_f1']:.4f}"
        )

    print("\nSaved summary to:", summary_path)


if __name__ == "__main__":
    main()