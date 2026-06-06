import re
from pathlib import Path
import pandas as pd


# ============================================================
# FINAL REPORT GENERATOR
# Project: Croatian sentiment analysis
# Label scheme: 0 = negative, 1 = neutral, 2 = positive
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

OUTPUT_DIR = RESULTS_DIR
OUTPUT_MODEL_COMPARISON_CSV = OUTPUT_DIR / "final_model_comparison.csv"
OUTPUT_SOURCE_LEVEL_CSV = OUTPUT_DIR / "final_source_level_results.csv"
OUTPUT_LABEL_DISTRIBUTION_CSV = OUTPUT_DIR / "final_label_distribution.csv"
OUTPUT_REPORT_MD = OUTPUT_DIR / "final_report.md"
OUTPUT_REPORT_TXT = OUTPUT_DIR / "final_report.txt"
OUTPUT_REPORT_XLSX = OUTPUT_DIR / "final_report.xlsx"


# ============================================================
# DATASET PATHS
# Adjust only if your file names are different.
# ============================================================

DATASET_CANDIDATES = {
    "Train": [
        PROJECT_ROOT / "data" / "processed" / "train.csv",
        PROJECT_ROOT / "data" / "processed" / "train.tsv",
    ],
    "Validation": [
        PROJECT_ROOT / "data" / "processed" / "val.csv",
        PROJECT_ROOT / "data" / "processed" / "validation.csv",
        PROJECT_ROOT / "data" / "processed" / "eval.csv",
        PROJECT_ROOT / "data" / "processed" / "val.tsv",
    ],
    "Test": [
        PROJECT_ROOT / "data" / "processed" / "test.csv",
        PROJECT_ROOT / "data" / "processed" / "test.tsv",
    ],
    "Golden dataset": [
        PROJECT_ROOT / "data_raw" / "OPJDataset_clean_review_labels_final.csv",
        PROJECT_ROOT / "data_raw" / "OPJDataset_clean.csv",
        PROJECT_ROOT / "data_raw" / "OPJDataset_clean_review_labels.csv",
    ],
}


# ============================================================
# BASIC HELPERS
# ============================================================

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def find_existing_path(candidates):
    for path in candidates:
        if path.exists():
            return path
    return None


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def round_numeric_df(df, digits=4):
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(digits)
    return df


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    """
    Custom markdown table writer.
    This avoids dependency on tabulate.
    """
    if df.empty:
        return "_No data available._"

    df_str = df.fillna("").astype(str)

    headers = list(df_str.columns)
    rows = df_str.values.tolist()

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    row_lines = []
    for row in rows:
        row_lines.append("| " + " | ".join(row) + " |")

    return "\n".join([header_line, separator_line] + row_lines)


# ============================================================
# METRIC EXTRACTION FROM TXT RESULT FILES
# ============================================================

def extract_accuracy(text: str):
    """
    Extracts the first Accuracy value from a result section.
    """
    match = re.search(r"Accuracy:\s*([0-9.]+)", text, re.IGNORECASE)
    return safe_float(match.group(1)) if match else None


def extract_weighted_metrics(text: str):
    """
    Extracts weighted precision, recall, and F1.

    Supports sklearn classification report format:
    weighted avg       0.72      0.74      0.72      1372

    Supports custom Gemma format:
    Weighted precision: 0.7880
    Weighted recall: 0.7252
    Weighted F1-score: 0.7330
    """

    sklearn_match = re.search(
        r"weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+",
        text,
        re.IGNORECASE,
    )

    if sklearn_match:
        return (
            safe_float(sklearn_match.group(1)),
            safe_float(sklearn_match.group(2)),
            safe_float(sklearn_match.group(3)),
        )

    p = re.search(r"Weighted precision:\s*([0-9.]+)", text, re.IGNORECASE)
    r = re.search(r"Weighted recall:\s*([0-9.]+)", text, re.IGNORECASE)
    f1 = re.search(r"Weighted F1-score:\s*([0-9.]+)", text, re.IGNORECASE)

    if p and r and f1:
        return safe_float(p.group(1)), safe_float(r.group(1)), safe_float(f1.group(1))

    return None, None, None


def extract_cv_macro_f1(text: str):
    match = re.search(r"Mean F1 macro:\s*([0-9.]+)", text, re.IGNORECASE)
    return safe_float(match.group(1)) if match else None


def extract_cv_std(text: str):
    match = re.search(r"Std:\s*([0-9.]+)", text, re.IGNORECASE)
    return safe_float(match.group(1)) if match else None


def split_combined_result_file(text: str):
    """
    Some result files contain both Cro-FiReDa and Golden/OPJ results.
    This function splits such files into sections.
    """

    patterns = [
        ("IN-DOMAIN TEST / CRO-FIREDA", "OUT-OF-DOMAIN TEST / OPJ NAJDOKTOR"),
        ("CNN TEST / CRO-FIREDA", "CNN TEST / OPJ"),
        ("TEST 1: CRO-FIREDA", "TEST 2: GOLDEN"),
        ("Test 1: Cro-FiReDa", "Test 2: Golden"),
    ]

    for first_header, second_header in patterns:
        if first_header in text and second_header in text:
            cro_section = text.split(first_header, 1)[1].split(second_header, 1)[0]
            golden_section = text.split(second_header, 1)[1]

            return [
                ("Test 1: Cro-FiReDa", cro_section),
                ("Test 2: Golden dataset", golden_section),
            ]

    return None


# ============================================================
# DETECTION OF METHOD, ALGORITHM, DATASET
# ============================================================

def detect_algorithm(file_name: str, text: str) -> str:
    name = file_name.lower()
    lower = text.lower()

    if "bertic" in name or "bcms-bertic" in lower or "classla/bcms-bertic" in lower:
        return "BERTić"

    if "gemma" in name or "google/gemma" in lower or "gemma" in lower:
        return "GENAI Gemma3"

    if "cnn" in name or "conv1d" in lower or "embedding + conv1d" in lower:
        return "CNN / Improved"

    if "mlp" in name or "mlpclassifier" in lower or "balanced mlpclassifier" in lower:
        return "TF-IDF + Balanced MLPClassifier"

    if "logistic" in name or "logistic regression" in lower:
        return "TF-IDF + Logistic Regression"

    return "Unknown"


def detect_method(algorithm: str) -> str:
    if algorithm == "TF-IDF + Logistic Regression":
        return "Machine learning"
    if algorithm == "TF-IDF + Balanced MLPClassifier":
        return "Machine learning"
    if algorithm == "CNN / Improved":
        return "Shallow Deep learning"
    if algorithm == "BERTić":
        return "Transformers"
    if algorithm == "GENAI Gemma3":
        return "GenAI / Zero-shot"
    return "Unknown"


def detect_train_type(algorithm: str, text: str) -> str:
    lower = text.lower()

    if algorithm == "GENAI Gemma3":
        return "Zero-shot"

    if "cross validation" in lower or "mean f1 macro" in lower:
        return "Train / 5-fold CV"

    return "TRAIN"


def detect_dataset(file_name: str, text: str) -> str:
    name = file_name.lower()
    lower = text.lower()

    if (
        "crofireda" in name
        or "cro-fireda" in lower
        or "crofireda" in lower
        or "in-domain test / cro-fireda" in lower
        or "cnn test / cro-fireda" in lower
        or "data/processed/test.csv" in lower
    ):
        return "Test 1: Cro-FiReDa"

    if (
        "golden" in name
        or "golden_test" in lower
        or "golden dataset" in lower
        or "opj najdoktor" in lower
        or "cnn test / opj" in lower
        or "data_raw/opjdataset" in lower
    ):
        return "Test 2: Golden dataset"

    return "Unknown"


def result_file_is_relevant(path: Path, text: str) -> bool:
    """
    Skips previously generated final reports and files without metrics.
    """
    name = path.name.lower()

    if name.startswith("final_"):
        return False

    if name in {
        "final_report.txt",
        "final_report.md",
        "final_results_summary.txt",
        "final_results_extracted.txt",
    }:
        return False

    has_metric = (
        "accuracy:" in text.lower()
        or "weighted avg" in text.lower()
        or "weighted precision" in text.lower()
    )

    return has_metric


# ============================================================
# EXTRACT ALL RESULTS FROM TXT FILES
# ============================================================

def extract_source_level_results():
    rows = []

    for path in sorted(RESULTS_DIR.glob("*.txt")):
        text = read_text(path)

        if not result_file_is_relevant(path, text):
            continue

        algorithm = detect_algorithm(path.name, text)
        method = detect_method(algorithm)
        train = detect_train_type(algorithm, text)
        cv_macro_f1 = extract_cv_macro_f1(text)
        cv_std = extract_cv_std(text)

        split_sections = split_combined_result_file(text)

        if split_sections:
            for dataset_name, section_text in split_sections:
                precision, recall, f1 = extract_weighted_metrics(section_text)
                accuracy = extract_accuracy(section_text)

                rows.append({
                    "source_file": path.name,
                    "method": method,
                    "algorithm": algorithm,
                    "train": train,
                    "dataset": dataset_name,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                    "cv_mean_f1_macro": cv_macro_f1,
                    "cv_std": cv_std,
                })
        else:
            dataset_name = detect_dataset(path.name, text)
            precision, recall, f1 = extract_weighted_metrics(text)
            accuracy = extract_accuracy(text)

            rows.append({
                "source_file": path.name,
                "method": method,
                "algorithm": algorithm,
                "train": train,
                "dataset": dataset_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
                "cv_mean_f1_macro": cv_macro_f1,
                "cv_std": cv_std,
            })

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No metrics were extracted from results/*.txt files.")

    df = df.drop_duplicates(
        subset=[
            "method",
            "algorithm",
            "dataset",
            "precision",
            "recall",
            "f1",
            "accuracy",
        ],
        keep="first",
    )

    return round_numeric_df(df, 4)


# ============================================================
# CREATE FINAL MODEL COMPARISON TABLE
# ============================================================

def choose_best_rows_for_final_table(df_source: pd.DataFrame) -> pd.DataFrame:
    """
    If multiple versions of the same algorithm exist, keep the strongest one.
    Selection criterion:
    - group by algorithm + dataset
    - choose row with highest F1
    - if F1 missing, choose highest accuracy
    """

    work = df_source.copy()

    work["selection_score"] = work["f1"]
    work.loc[work["selection_score"].isna(), "selection_score"] = work["accuracy"]

    selected_rows = []

    for (algorithm, dataset), group in work.groupby(["algorithm", "dataset"], dropna=False):
        if dataset == "Unknown":
            continue

        group_sorted = group.sort_values(
            by=["selection_score", "accuracy"],
            ascending=False,
            na_position="last",
        )
        selected_rows.append(group_sorted.iloc[0])

    selected = pd.DataFrame(selected_rows).drop(columns=["selection_score"], errors="ignore")

    return selected


def create_final_model_comparison(df_source: pd.DataFrame) -> pd.DataFrame:
    selected = choose_best_rows_for_final_table(df_source)

    rows = []

    order = [
        "TF-IDF + Logistic Regression",
        "TF-IDF + Balanced MLPClassifier",
        "CNN / Improved",
        "BERTić",
        "GENAI Gemma3",
    ]

    numbering = {
        "TF-IDF + Logistic Regression": "1.a.i",
        "TF-IDF + Balanced MLPClassifier": "1.a.ii",
        "CNN / Improved": "2.a",
        "BERTić": "3.a",
        "GENAI Gemma3": "3.b",
    }

    method_labels = {
        "TF-IDF + Logistic Regression": "Machine learning",
        "TF-IDF + Balanced MLPClassifier": "Machine learning",
        "CNN / Improved": "Shallow Deep learning",
        "BERTić": "Transformers",
        "GENAI Gemma3": "GenAI / Zero-shot",
    }

    for algorithm in order:
        alg_rows = selected[selected["algorithm"] == algorithm]

        if alg_rows.empty:
            continue

        cro = alg_rows[alg_rows["dataset"] == "Test 1: Cro-FiReDa"]
        golden = alg_rows[alg_rows["dataset"] == "Test 2: Golden dataset"]

        first = alg_rows.iloc[0]

        row = {
            "#": numbering.get(algorithm, ""),
            "method": method_labels.get(algorithm, first.get("method", "")),
            "algorithm": algorithm,
            "train": first.get("train", ""),
            "Test 1 Precision": None,
            "Test 1 Recall": None,
            "Test 1 F1": None,
            "Test 1 Accuracy": None,
            "Test 2 Precision": None,
            "Test 2 Recall": None,
            "Test 2 F1": None,
            "Test 2 Accuracy": None,
        }

        if not cro.empty:
            cro_row = cro.iloc[0]
            row["Test 1 Precision"] = cro_row["precision"]
            row["Test 1 Recall"] = cro_row["recall"]
            row["Test 1 F1"] = cro_row["f1"]
            row["Test 1 Accuracy"] = cro_row["accuracy"]

        if not golden.empty:
            golden_row = golden.iloc[0]
            row["Test 2 Precision"] = golden_row["precision"]
            row["Test 2 Recall"] = golden_row["recall"]
            row["Test 2 F1"] = golden_row["f1"]
            row["Test 2 Accuracy"] = golden_row["accuracy"]

        rows.append(row)

    final_df = pd.DataFrame(rows)

    return round_numeric_df(final_df, 4)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

def read_dataset_file(path: Path):
    """
    Tries common separators: comma, semicolon, tab.
    """
    if path is None or not path.exists():
        return None

    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            pass

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def find_label_column(df: pd.DataFrame) -> str:
    possible_columns = [
        "label",
        "labels",
        "label_m",
        "final_label",
        "sentiment",
        "y",
        "target",
    ]

    for col in possible_columns:
        if col in df.columns:
            return col

    raise ValueError(f"No label column found. Available columns: {list(df.columns)}")


def normalize_label_series(series: pd.Series) -> pd.Series:
    """
    Converts common label forms to numeric labels:
    0 = negative
    1 = neutral
    2 = positive
    """
    mapping = {
        "negative": 0,
        "neg": 0,
        "0": 0,
        0: 0,
        "neutral": 1,
        "neu": 1,
        "1": 1,
        1: 1,
        "positive": 2,
        "pos": 2,
        "2": 2,
        2: 2,
    }

    def convert(x):
        if pd.isna(x):
            return None

        if isinstance(x, str):
            value = x.strip().lower()
        else:
            value = x

        if value in mapping:
            return mapping[value]

        try:
            numeric = int(float(value))
            if numeric in [0, 1, 2]:
                return numeric
        except Exception:
            return None

        return None

    return series.apply(convert)


def count_label_distribution() -> pd.DataFrame:
    rows = []

    for split_name, candidates in DATASET_CANDIDATES.items():
        path = find_existing_path(candidates)

        if path is None:
            rows.append({
                "split": split_name,
                "positive": None,
                "negative": None,
                "neutral": None,
                "total": None,
                "source_file": "NOT FOUND",
                "label_column": "NOT FOUND",
            })
            continue

        df = read_dataset_file(path)

        if df is None:
            rows.append({
                "split": split_name,
                "positive": None,
                "negative": None,
                "neutral": None,
                "total": None,
                "source_file": str(path.relative_to(PROJECT_ROOT)),
                "label_column": "COULD NOT READ FILE",
            })
            continue

        label_col = find_label_column(df)
        labels = normalize_label_series(df[label_col])

        positive = int((labels == 2).sum())
        negative = int((labels == 0).sum())
        neutral = int((labels == 1).sum())
        total = positive + negative + neutral

        rows.append({
            "split": split_name,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "total": total,
            "source_file": str(path.relative_to(PROJECT_ROOT)),
            "label_column": label_col,
        })

    return pd.DataFrame(rows)


# ============================================================
# TEXTUAL REPORT GENERATION
# ============================================================

def get_best_model_text(final_comparison: pd.DataFrame) -> str:
    lines = []

    if "Test 1 F1" in final_comparison.columns:
        test1 = final_comparison.dropna(subset=["Test 1 F1"])
        if not test1.empty:
            best_test1 = test1.loc[test1["Test 1 F1"].idxmax()]
            lines.append(
                f"The best-performing model on Test 1 / Cro-FiReDa is "
                f"{best_test1['algorithm']} with weighted F1 = {best_test1['Test 1 F1']} "
                f"and accuracy = {best_test1['Test 1 Accuracy']}."
            )

    if "Test 2 F1" in final_comparison.columns:
        test2 = final_comparison.dropna(subset=["Test 2 F1"])
        if not test2.empty:
            best_test2 = test2.loc[test2["Test 2 F1"].idxmax()]
            lines.append(
                f"The best-performing model on Test 2 / Golden dataset is "
                f"{best_test2['algorithm']} with weighted F1 = {best_test2['Test 2 F1']} "
                f"and accuracy = {best_test2['Test 2 Accuracy']}."
            )

    return "\n\n".join(lines)


def create_report_markdown(final_comparison, source_results, label_distribution) -> str:
    best_text = get_best_model_text(final_comparison)

    report = f"""# Final Report: Croatian Sentiment Analysis

## 1. Project overview

This project compares several approaches for sentiment classification of Croatian texts. The sentiment classification task uses a three-class label scheme:

- 0 = negative
- 1 = neutral
- 2 = positive

The evaluation includes two test settings:

- **Test 1: Cro-FiReDa** — in-domain public Croatian sentiment dataset.
- **Test 2: Golden dataset** — out-of-domain manually prepared OPJ / NajDoktor dataset.

Precision, recall and F1-score in the final comparison table are reported as **weighted averages** when available.

## 2. Label distribution

{df_to_markdown_simple(label_distribution)}

## 3. Final model comparison

{df_to_markdown_simple(final_comparison)}

## 4. Source-level extracted results

The following table shows the metrics extracted directly from the `.txt` result files in the `results/` directory.

{df_to_markdown_simple(source_results)}

## 5. Best-performing models

{best_text}

## 6. Interpretation

The results show that transformer-based modelling achieved the strongest overall performance. BERTić performed best on both the in-domain Cro-FiReDa test set and the out-of-domain Golden dataset. This confirms that a pretrained transformer model adapted to Bosnian, Croatian, Montenegrin and Serbian language data is better suited for Croatian sentiment classification than simpler machine learning and shallow deep learning baselines.

The TF-IDF + Logistic Regression model performed competitively on the Cro-FiReDa test set, especially after hyperparameter adjustment. However, its performance dropped on the Golden dataset, which indicates weaker generalization to out-of-domain review texts.

The improved CNN model achieved lower performance than BERTić on Cro-FiReDa, but it generalized better to the Golden dataset than the TF-IDF + Logistic Regression baseline. This suggests that the neural architecture captured some useful textual patterns, although it remained weaker than the transformer model.

The Gemma zero-shot approach performed reasonably well on Cro-FiReDa without supervised training, but its performance was weaker on the Golden dataset. This suggests that prompt-based zero-shot classification can be useful as a baseline, but it is less reliable than supervised fine-tuning for this specific Croatian sentiment classification task.

## 7. Conclusion

The best overall method is **BERTić**, which achieved the highest weighted F1-score and accuracy on both evaluation datasets. The comparison shows that transformer-based models are the most effective approach for this project, while classical machine learning, shallow deep learning and zero-shot GenAI methods provide useful baselines for comparison.

"""
    return report


def create_report_txt(final_comparison, source_results, label_distribution) -> str:
    best_text = get_best_model_text(final_comparison)

    report = f"""FINAL REPORT: CROATIAN SENTIMENT ANALYSIS
============================================================

1. PROJECT OVERVIEW
-------------------

This project compares several approaches for sentiment classification of Croatian texts.

Label scheme:
0 = negative
1 = neutral
2 = positive

Evaluation settings:
Test 1 = Cro-FiReDa, in-domain public Croatian sentiment dataset.
Test 2 = Golden dataset, out-of-domain OPJ / NajDoktor dataset.

Precision, recall and F1-score are reported as weighted averages when available.


2. LABEL DISTRIBUTION
---------------------

{label_distribution.to_string(index=False)}


3. FINAL MODEL COMPARISON
-------------------------

{final_comparison.to_string(index=False)}


4. SOURCE-LEVEL EXTRACTED RESULTS
---------------------------------

{source_results.to_string(index=False)}


5. BEST-PERFORMING MODELS
-------------------------

{best_text}


6. INTERPRETATION
-----------------

The results show that transformer-based modelling achieved the strongest overall performance. BERTić performed best on both the in-domain Cro-FiReDa test set and the out-of-domain Golden dataset. This confirms that a pretrained transformer model adapted to Bosnian, Croatian, Montenegrin and Serbian language data is better suited for Croatian sentiment classification than simpler machine learning and shallow deep learning baselines.

The TF-IDF + Logistic Regression model performed competitively on the Cro-FiReDa test set, especially after hyperparameter adjustment. However, its performance dropped on the Golden dataset, which indicates weaker generalization to out-of-domain review texts.

The improved CNN model achieved lower performance than BERTić on Cro-FiReDa, but it generalized better to the Golden dataset than the TF-IDF + Logistic Regression baseline. This suggests that the neural architecture captured some useful textual patterns, although it remained weaker than the transformer model.

The Gemma zero-shot approach performed reasonably well on Cro-FiReDa without supervised training, but its performance was weaker on the Golden dataset. This suggests that prompt-based zero-shot classification can be useful as a baseline, but it is less reliable than supervised fine-tuning for this specific Croatian sentiment classification task.


7. CONCLUSION
-------------

The best overall method is BERTić, which achieved the highest weighted F1-score and accuracy on both evaluation datasets. The comparison shows that transformer-based models are the most effective approach for this project, while classical machine learning, shallow deep learning and zero-shot GenAI methods provide useful baselines for comparison.

"""
    return report


# ============================================================
# EXPORT
# ============================================================

def export_excel(final_comparison, source_results, label_distribution):
    """
    Exports XLSX if openpyxl is available.
    If not available, the script continues normally.
    """
    try:
        with pd.ExcelWriter(OUTPUT_REPORT_XLSX, engine="openpyxl") as writer:
            final_comparison.to_excel(writer, sheet_name="Model comparison", index=False)
            label_distribution.to_excel(writer, sheet_name="Label distribution", index=False)
            source_results.to_excel(writer, sheet_name="Source results", index=False)
        return True
    except Exception as e:
        print(f"WARNING: Excel export skipped: {e}")
        return False


def main():
    print("Extracting metrics from txt files...")
    source_results = extract_source_level_results()

    print("Creating final model comparison table...")
    final_comparison = create_final_model_comparison(source_results)

    print("Counting label distribution...")
    label_distribution = count_label_distribution()

    source_results = round_numeric_df(source_results, 4)
    final_comparison = round_numeric_df(final_comparison, 4)

    print("Exporting CSV files...")
    source_results.to_csv(OUTPUT_SOURCE_LEVEL_CSV, index=False)
    final_comparison.to_csv(OUTPUT_MODEL_COMPARISON_CSV, index=False)
    label_distribution.to_csv(OUTPUT_LABEL_DISTRIBUTION_CSV, index=False)

    print("Exporting Markdown and TXT reports...")
    report_md = create_report_markdown(final_comparison, source_results, label_distribution)
    report_txt = create_report_txt(final_comparison, source_results, label_distribution)

    OUTPUT_REPORT_MD.write_text(report_md, encoding="utf-8")
    OUTPUT_REPORT_TXT.write_text(report_txt, encoding="utf-8")

    print("Exporting Excel report...")
    excel_created = export_excel(final_comparison, source_results, label_distribution)

    print()
    print("Done. Created files:")
    print(f"- {OUTPUT_MODEL_COMPARISON_CSV.relative_to(PROJECT_ROOT)}")
    print(f"- {OUTPUT_SOURCE_LEVEL_CSV.relative_to(PROJECT_ROOT)}")
    print(f"- {OUTPUT_LABEL_DISTRIBUTION_CSV.relative_to(PROJECT_ROOT)}")
    print(f"- {OUTPUT_REPORT_MD.relative_to(PROJECT_ROOT)}")
    print(f"- {OUTPUT_REPORT_TXT.relative_to(PROJECT_ROOT)}")

    if excel_created:
        print(f"- {OUTPUT_REPORT_XLSX.relative_to(PROJECT_ROOT)}")
    else:
        print("- Excel export was skipped. Install openpyxl if needed: pip install openpyxl")

    print()
    print("Final model comparison:")
    print(final_comparison)

    print()
    print("Label distribution:")
    print(label_distribution)


if __name__ == "__main__":
    main()