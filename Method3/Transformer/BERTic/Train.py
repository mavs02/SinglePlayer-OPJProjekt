import os
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


MODEL_NAME = "classla/bcms-bertic"

TRAIN_PATH = "data/processed/train.csv"
VAL_PATH = "data/processed/val.csv"

OUTPUT_DIR = "Method3/Transformer/BERTic/bertic_model"
RESULTS_DIR = "results"


def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"Column not found. Tried: {possible_names}. Existing columns: {list(df.columns)}")


def load_data(path):
    df = pd.read_csv(path)

    text_col = find_column(df, ["text", "sentence", "review", "content"])
    label_col = find_column(df, ["sentiment", "label", "labels"])

    df = df[[text_col, label_col]].dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)

    return df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(labels, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    train_df = load_data(TRAIN_PATH)
    val_df = load_data(VAL_PATH)

    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Train label distribution:")
    print(train_df["label"].value_counts())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    train_dataset = train_dataset.remove_columns(["text"])
    val_dataset = val_dataset.remove_columns(["text"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="Method3/Transformer/BERTic/logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training BERTic model...")
    trainer.train()

    print("Evaluating on validation set...")
    metrics = trainer.evaluate()

    print(metrics)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open("results/Trial_4_BERTic.txt", "w", encoding="utf-8") as f:
        f.write("Method 3: Transformer - BERTic\n")
        f.write("Model: classla/bcms-bertic\n")
        f.write("Train data: data/processed/train.csv\n")
        f.write("Validation data: data/processed/val.csv\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    print("BERTic model and results saved.")


if __name__ == "__main__":
    main()
