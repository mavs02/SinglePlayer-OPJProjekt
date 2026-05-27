import os
import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

# ============================================================
# PURPOSE
# ============================================================
# This script:
# 1. calculates Fleiss' Kappa between AI chatbot labels
# 2. calculates Fleiss' Kappa between manual review label and AI labels
# 3. creates the final sentiment label using majority voting
#
# Important:
# - AI labels are review-level labels.
# - label_m is sentence-level.
# - For agreement, label_m is converted to label_m_review.
# - Final label is created on sentence level.

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "data_raw/OPJDataset_clean_review_labels.csv"

RESULTS_TXT_PATH = "results/fleiss_kappa.txt"
RESULTS_CSV_PATH = "results/fleiss_kappa_summary.csv"

OUTPUT_DATASET_PATH = "data_raw/OPJDataset_clean_review_labels_final.csv"
OUTPUT_REVIEW_AGREEMENT_PATH = "data_raw/OPJDataset_review_agreement_labels.csv"

# ============================================================
# REVIEW IDENTIFICATION
# ============================================================

# One review is identified by title + review_id.
KEY_COLUMNS = ["title", "review_id"]

# ============================================================
# LABEL COLUMNS
# ============================================================

AI_COLUMNS = [
    "label_chatgpt_review",
    "label_perplexity_review",
    "label_copilot_review",
]

ALL_REVIEW_COLUMNS = [
    "label_m_review",
    "label_chatgpt_review",
    "label_perplexity_review",
    "label_copilot_review",
]

# ============================================================
# FUNCTIONS
# ============================================================

def interpret_kappa(kappa):
    """Return standard interpretation of Fleiss' Kappa."""
    if kappa < 0:
        return "No agreement"
    elif kappa <= 0.20:
        return "Slight agreement"
    elif kappa <= 0.40:
        return "Fair agreement"
    elif kappa <= 0.60:
        return "Moderate agreement"
    elif kappa <= 0.80:
        return "Substantial agreement"
    else:
        return "Almost perfect agreement"


def majority_label_from_series(labels):
    """
    Convert sentence-level manual labels into one review-level label.

    Example:
    [2, 2, 1, 2] -> 2
    """
    labels = pd.to_numeric(labels, errors="coerce").dropna().astype(int)

    if len(labels) == 0:
        return None

    counts = labels.value_counts()
    max_count = counts.max()
    top_labels = counts[counts == max_count].index.tolist()

    # If there is a tie and neutral is included, choose neutral.
    if 1 in top_labels:
        return 1

    return int(top_labels[0])


def create_final_sentence_label(row):
    """
    Create final sentence-level label.

    Sources:
    - label_m
    - label_chatgpt_review
    - label_perplexity_review
    - label_copilot_review

    Rule:
    - clear majority -> majority label
    - tie -> label_m wins
    """
    labels = [
        row["label_m"],
        row["label_chatgpt_review"],
        row["label_perplexity_review"],
        row["label_copilot_review"],
    ]

    labels = [int(label) for label in labels if pd.notna(label)]

    if not labels:
        return None

    counts = pd.Series(labels).value_counts()
    max_count = counts.max()
    top_labels = counts[counts == max_count].index.tolist()

    if len(top_labels) == 1:
        return int(top_labels[0])

    return int(row["label_m"])


def calculate_fleiss_kappa(df, columns):
    """
    Calculate Fleiss' Kappa for selected columns.
    Each row must be one annotated item.
    Here, each row is one review.
    """
    missing_columns = [column for column in columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}. "
            f"Existing columns: {list(df.columns)}"
        )

    ratings_df = df[columns].dropna()

    if ratings_df.empty:
        raise ValueError(f"No complete rows for columns: {columns}")

    ratings_raw = ratings_df.astype(int).to_numpy()
    ratings_matrix, categories = aggregate_raters(ratings_raw)

    kappa = fleiss_kappa(ratings_matrix, method="fleiss")

    return {
        "samples": len(ratings_df),
        "categories": [int(category) for category in categories],
        "kappa": float(kappa),
        "interpretation": interpret_kappa(kappa),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs("results", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "First create OPJDataset_clean_review_labels.csv."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = KEY_COLUMNS + [
        "sentence_id",
        "label_m",
        "label_chatgpt_review",
        "label_perplexity_review",
        "label_copilot_review",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # ========================================================
    # CREATE REVIEW-LEVEL AGREEMENT DATASET
    # ========================================================

    review_df = (
        df.groupby(KEY_COLUMNS, sort=False)
        .agg(
            label_m_review=("label_m", majority_label_from_series),
            label_chatgpt_review=("label_chatgpt_review", "first"),
            label_perplexity_review=("label_perplexity_review", "first"),
            label_copilot_review=("label_copilot_review", "first"),
            number_of_sentences=("sentence_id", "count"),
        )
        .reset_index()
    )

    review_df.to_csv(
        OUTPUT_REVIEW_AGREEMENT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # CALCULATE AGREEMENT
    # ========================================================

    ai_result = calculate_fleiss_kappa(review_df, AI_COLUMNS)
    all_result = calculate_fleiss_kappa(review_df, ALL_REVIEW_COLUMNS)

    # ========================================================
    # CREATE FINAL SENTENCE-LEVEL LABEL
    # ========================================================

    df["label_final_after_agreement"] = df.apply(
        create_final_sentence_label,
        axis=1
    )

    df["label"] = df["label_final_after_agreement"]

    df.to_csv(
        OUTPUT_DATASET_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    summary_df = pd.DataFrame([
        {
            "comparison": "AI chatbots only",
            "level": "review",
            "samples": ai_result["samples"],
            "kappa": round(ai_result["kappa"], 4),
            "interpretation": ai_result["interpretation"],
        },
        {
            "comparison": "Manual review label + AI chatbots",
            "level": "review",
            "samples": all_result["samples"],
            "kappa": round(all_result["kappa"], 4),
            "interpretation": all_result["interpretation"],
        },
    ])

    summary_df.to_csv(
        RESULTS_CSV_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # SHORT TXT OUTPUT
    # ========================================================

    label_counts = df["label_final_after_agreement"].value_counts().sort_index()

    count_0 = int(label_counts.get(0, 0))
    count_1 = int(label_counts.get(1, 0))
    count_2 = int(label_counts.get(2, 0))

    output = f"""FLEISS KAPPA SUMMARY

Dataset: {DATA_PATH}
Sentence rows: {len(df)}
Review rows: {len(review_df)}

AGREEMENT RESULTS

Comparison                          Samples   Kappa    Interpretation
AI chatbots only                    {ai_result["samples"]:<9} {ai_result["kappa"]:.4f}   {ai_result["interpretation"]}
Manual review label + AI chatbots   {all_result["samples"]:<9} {all_result["kappa"]:.4f}   {all_result["interpretation"]}

FINAL LABEL DISTRIBUTION

Label   Meaning    Count
0       Negative   {count_0}
1       Neutral    {count_1}
2       Positive   {count_2}

OUTPUT FILES

Summary:
{RESULTS_CSV_PATH}

Review-level agreement dataset:
{OUTPUT_REVIEW_AGREEMENT_PATH}

Final sentence-level dataset:
{OUTPUT_DATASET_PATH}

FINAL LABEL RULE

Majority vote:
label_m + label_chatgpt_review + label_perplexity_review + label_copilot_review

Tie rule:
label_m wins.
"""

    print(output)

    with open(RESULTS_TXT_PATH, "w", encoding="utf-8") as file:
        file.write(output)

    print("Saved:", RESULTS_TXT_PATH)
    print("Saved:", RESULTS_CSV_PATH)
    print("Saved:", OUTPUT_REVIEW_AGREEMENT_PATH)
    print("Saved:", OUTPUT_DATASET_PATH)


if __name__ == "__main__":
    main()