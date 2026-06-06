import os
import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


# ============================================================
# METHOD 4: GenAI - One-shot sentiment classification
# ============================================================

INPUT_CSV = "data_raw/OPJDataset_clean_review_labels.csv"

OUTPUT_CSV = "results/method4_genai_one_shot_results.csv"
METRICS_TXT = "results/method4_genai_one_shot_metrics.txt"

MODEL_NAME = "google/gemma-3-1b-it"

# Ako znaš točan naziv stupaca, možeš ih direktno upisati ovdje.
# Ako ostaviš None, skripta će pokušati sama pronaći stupce.
TEXT_COLUMN = None
LABEL_COLUMN = None

TEXT_COLUMN_CANDIDATES = [
    "text",
    "review",
    "sentence",
    "comment",
    "recenzija",
    "Review",
    "Text"
]

LABEL_COLUMN_CANDIDATES = [
    "label",
    "sentiment",
    "manual_label",
    "human_label",
    "gold_label",
    "Label",
    "Sentiment"
]

# Ako želiš ručno odabrati primjer iz dataseta, ovdje upiši index retka.
# Primjer:
# ONE_SHOT_EXAMPLE_INDEX = 15
#
# Ako ostane None, skripta će sama uzeti jedan neutralni primjer ako postoji.
ONE_SHOT_EXAMPLE_INDEX = None

# 0 = negativno, 1 = neutralno, 2 = pozitivno
PREFERRED_EXAMPLE_LABEL = 1


def find_column(df, explicit_column, candidates, column_type):
    if explicit_column is not None:
        if explicit_column not in df.columns:
            raise ValueError(
                f"Stupac '{explicit_column}' ne postoji u CSV-u. "
                f"Dostupni stupci su: {list(df.columns)}"
            )
        return explicit_column

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Nisam pronašla {column_type} stupac. "
        f"Očekivani nazivi: {candidates}. "
        f"Dostupni stupci su: {list(df.columns)}"
    )


def normalize_label(value):
    """
    Pretvara labelu u int 0, 1 ili 2.
    Radi i ako je labela spremljena kao string.
    """
    if pd.isna(value):
        return None

    value_str = str(value).strip()

    if value_str in ["0", "1", "2"]:
        return int(value_str)

    lowered = value_str.lower()

    if lowered in ["negative", "negativno", "negativan"]:
        return 0
    if lowered in ["neutral", "neutralno", "neutralan"]:
        return 1
    if lowered in ["positive", "pozitivno", "pozitivan"]:
        return 2

    return None


def clean_prediction(generated_answer):
    """
    Iz odgovora modela izvlači oznaku 0, 1 ili 2.
    Ako model vrati nešto nevažeće, vraća -1.
    """
    if generated_answer is None:
        return -1

    generated_answer = str(generated_answer).strip()

    match = re.search(r"\b[012]\b", generated_answer)
    if match:
        return int(match.group(0))

    return -1


def create_one_shot_prompt(review_text, example_text, example_label):
    prompt = f"""
Zadatak: Klasificiraj sentiment hrvatske recenzije.

Koristi isključivo jednu od sljedećih oznaka:
0 = negativan sentiment
1 = neutralan sentiment
2 = pozitivan sentiment

Primjer:
Recenzija: "{example_text}"
Oznaka: {example_label}

Sada klasificiraj sljedeću recenziju.
Odgovori isključivo jednim brojem: 0, 1 ili 2.

Recenzija: "{review_text}"
Oznaka:
"""
    return prompt.strip()


def generate_prediction(review_text, example_text, example_label, tokenizer, model):
    prompt = create_one_shot_prompt(
        review_text=review_text,
        example_text=example_text,
        example_label=example_label
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Model često vrati cijeli prompt + odgovor.
    # Zato uzimamo samo dio nakon zadnje pojave "Oznaka:"
    answer_part = decoded.split("Oznaka:")[-1].strip()

    prediction = clean_prediction(answer_part)

    return prediction, answer_part


def select_one_shot_example(df, text_col, label_col):
    """
    Odabire jedan primjer iz dataseta za one-shot prompt.
    Ako je ONE_SHOT_EXAMPLE_INDEX zadan, koristi taj redak.
    Inače preferira neutralni primjer.
    """
    if ONE_SHOT_EXAMPLE_INDEX is not None:
        if ONE_SHOT_EXAMPLE_INDEX not in df.index:
            raise ValueError(
                f"Index {ONE_SHOT_EXAMPLE_INDEX} ne postoji u datasetu."
            )

        row = df.loc[ONE_SHOT_EXAMPLE_INDEX]
        example_text = str(row[text_col])
        example_label = normalize_label(row[label_col])

        if example_label not in [0, 1, 2]:
            raise ValueError(
                f"One-shot primjer ima nevažeću oznaku: {row[label_col]}"
            )

        return example_text, example_label, ONE_SHOT_EXAMPLE_INDEX

    df_valid = df.copy()
    df_valid["_normalized_label"] = df_valid[label_col].apply(normalize_label)
    df_valid = df_valid[df_valid["_normalized_label"].isin([0, 1, 2])]

    preferred = df_valid[df_valid["_normalized_label"] == PREFERRED_EXAMPLE_LABEL]

    if len(preferred) > 0:
        row = preferred.sample(1, random_state=42).iloc[0]
    else:
        row = df_valid.sample(1, random_state=42).iloc[0]

    example_text = str(row[text_col])
    example_label = int(row["_normalized_label"])
    example_index = row.name

    return example_text, example_label, example_index


def main():
    os.makedirs("results", exist_ok=True)

    print("=" * 60)
    print("METHOD 4 - GenAI One-shot Sentiment Classification")
    print("=" * 60)

    print("\nLoading dataset...")
    df = pd.read_csv(INPUT_CSV)

    text_col = find_column(
        df=df,
        explicit_column=TEXT_COLUMN,
        candidates=TEXT_COLUMN_CANDIDATES,
        column_type="tekstualni"
    )

    label_col = find_column(
        df=df,
        explicit_column=LABEL_COLUMN,
        candidates=LABEL_COLUMN_CANDIDATES,
        column_type="label"
    )

    print(f"Input CSV: {INPUT_CSV}")
    print(f"Text column: {text_col}")
    print(f"Label column: {label_col}")
    print(f"Number of rows: {len(df)}")

    df["_gold_label_normalized"] = df[label_col].apply(normalize_label)

    invalid_gold = df[~df["_gold_label_normalized"].isin([0, 1, 2])]
    if len(invalid_gold) > 0:
        print(f"\nWarning: {len(invalid_gold)} rows have invalid gold labels.")
        print("They will be ignored during evaluation.")

    example_text, example_label, example_index = select_one_shot_example(
        df=df,
        text_col=text_col,
        label_col=label_col
    )

    print("\nSelected one-shot example:")
    print(f"Index: {example_index}")
    print(f"Label: {example_label}")
    print(f"Text: {example_text}")

    print("\nLoading GenAI model...")
    print(f"Model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    predictions = []
    raw_outputs = []

    print("\nRunning classification...")

    for i, row in df.iterrows():
        review_text = str(row[text_col])

        prediction, raw_answer = generate_prediction(
            review_text=review_text,
            example_text=example_text,
            example_label=example_label,
            tokenizer=tokenizer,
            model=model
        )

        predictions.append(prediction)
        raw_outputs.append(raw_answer)

        if (i + 1) % 25 == 0:
            print(f"Processed {i + 1}/{len(df)} reviews...")

    df["method4_genai_one_shot_label"] = predictions
    df["method4_genai_one_shot_raw_output"] = raw_outputs

    # Ukloni pomoćni stupac ako ne želiš da ostane u CSV-u
    # ali ga ostavljam jer je koristan za provjeru evaluacije.
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nResults saved to: {OUTPUT_CSV}")

    valid_eval_df = df[
        df["_gold_label_normalized"].isin([0, 1, 2])
        & df["method4_genai_one_shot_label"].isin([0, 1, 2])
    ].copy()

    invalid_predictions = len(df) - len(df[df["method4_genai_one_shot_label"].isin([0, 1, 2])])

    print("\nEvaluation data:")
    print(f"Valid predictions: {len(valid_eval_df)}")
    print(f"Invalid predictions: {invalid_predictions}")

    with open(METRICS_TXT, "w", encoding="utf-8") as f:
        f.write("METHOD 4 - GenAI One-shot Sentiment Classification\n")
        f.write("==================================================\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Input CSV: {INPUT_CSV}\n")
        f.write(f"Output CSV: {OUTPUT_CSV}\n")
        f.write(f"Number of rows: {len(df)}\n\n")

        f.write("One-shot example\n")
        f.write("----------------\n")
        f.write(f"Example index: {example_index}\n")
        f.write(f"Example label: {example_label}\n")
        f.write(f"Example text: {example_text}\n\n")

        f.write("Evaluation data\n")
        f.write("----------------\n")
        f.write(f"Valid predictions: {len(valid_eval_df)}\n")
        f.write(f"Invalid predictions: {invalid_predictions}\n\n")

        if len(valid_eval_df) > 0:
            y_true = valid_eval_df["_gold_label_normalized"].astype(int)
            y_pred = valid_eval_df["method4_genai_one_shot_label"].astype(int)

            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )

            report = classification_report(y_true, y_pred, zero_division=0)

            print("\nEvaluation results:")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print("\nClassification report:")
            print(report)

            f.write("Metrics\n")
            f.write("-------\n")
            f.write(f"Accuracy:  {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall:    {recall:.4f}\n")
            f.write(f"F1-score:  {f1:.4f}\n\n")

            f.write("Classification report\n")
            f.write("---------------------\n")
            f.write(report)
        else:
            f.write("No valid predictions available for evaluation.\n")

    print(f"\nMetrics saved to: {METRICS_TXT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
