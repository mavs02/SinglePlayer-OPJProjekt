import pandas as pd
from pathlib import Path

# ============================================================
# PURPOSE
# ============================================================
# This script uses the original sentence-level clean dataset
# and distributes review-level AI sentiment labels to sentences.
#
# Input dataset:
# - data_raw/OPJDataset_clean.csv
# - one row = one sentence
#
# The dataset already contains:
# - title: doctor/person name
# - review_id: comment number for that title
# - sentence_id: sentence number inside that comment
# - text: sentence text
# - label_m: manual sentence-level label
# - old AI labels: label_perplexity, label_copilot
# - old final label: label
#
# Goal:
# - keep label_m unchanged
# - remove old AI labels:
#     label_perplexity
#     label_copilot
#     label
# - add new AI labels that were assigned to the whole review:
#     label_chatgpt_review
#     label_perplexity_review
#     label_copilot_review
# - assign the same review-level AI label to every sentence
#   that belongs to the same comment
# - create a new final label using majority voting
#
# One unique comment is identified by:
# title + review_id
#
# This is because review_id starts again from 1 when a new title/name begins.


# ============================================================
# INPUTS
# ============================================================

ORIGINAL_DATASET_PATH = Path("data_raw/OPJDataset_clean.csv")

CHATGPT_LABELS_PATH = Path("data_raw/review_labels_chatgpt.txt")
PERPLEXITY_LABELS_PATH = Path("data_raw/review_labels_perplexity.txt")
COPILOT_LABELS_PATH = Path("data_raw/review_labels_copilot.txt")


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_PATH = Path("data_raw/OPJDataset_clean_review_labels.csv")


# ============================================================
# REVIEW IDENTIFICATION
# ============================================================

# These columns identify one complete comment/review.
KEY_COLUMNS = ["title", "review_id"]


# ============================================================
# LABEL MAPPING
# ============================================================
# Standard sentiment labels:
# 0 = negative
# 1 = neutral
# 2 = positive

LABEL_MAP = {
    "negativno": 0,
    "negative": 0,
    "neg": 0,
    "0": 0,

    "neutralno": 1,
    "neutral": 1,
    "neu": 1,
    "1": 1,

    "pozitivno": 2,
    "positive": 2,
    "poz": 2,
    "2": 2,
}


def normalize_label(value):
    """
    Converts textual labels into numeric labels.

    Examples:
    - pozitivno -> 2
    - neutralno -> 1
    - negativno -> 0
    """
    value = str(value).strip().lower()

    if value not in LABEL_MAP:
        raise ValueError(
            f"Unknown label: '{value}'. "
            "Allowed labels: negativno, neutralno, pozitivno, or 0, 1, 2."
        )

    return LABEL_MAP[value]


def read_label_file(path, expected_count):
    """
    Reads one .txt file with AI labels.

    The file must contain exactly one label per unique review.
    The number of labels must match the number of unique title + review_id pairs.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing label file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        raw_labels = [line.strip() for line in file.readlines() if line.strip()]

    numeric_labels = [normalize_label(label) for label in raw_labels]

    if len(numeric_labels) != expected_count:
        raise ValueError(
            f"File {path} contains {len(numeric_labels)} labels, "
            f"but the dataset contains {expected_count} unique reviews. "
            "The number of labels must match the number of unique comments."
        )

    return numeric_labels


def majority_vote(row):
    """
    Creates a new final label for each sentence.

    It uses four sources:
    - label_m: manual sentence-level label
    - label_chatgpt_review: ChatGPT label for the whole review
    - label_perplexity_review: Perplexity label for the whole review
    - label_copilot_review: Copilot label for the whole review

    If there is a clear majority, the majority label is used.
    If there is a tie, label_m is used because manual annotation has priority.
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

    # Clear majority
    if len(top_labels) == 1:
        return int(top_labels[0])

    # Tie -> manual label wins
    return int(row["label_m"])


def main():
    # --------------------------------------------------------
    # Load original clean dataset.
    # --------------------------------------------------------
    df = pd.read_csv(ORIGINAL_DATASET_PATH)

    print("Original sentence rows:", len(df))

    # --------------------------------------------------------
    # Check required columns.
    # --------------------------------------------------------
    required_columns = KEY_COLUMNS + ["sentence_id", "text", "label_m"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column in original dataset: {column}")

    # --------------------------------------------------------
    # Create a unique review table directly from the original dataset.
    # No separate review-level dataset is needed.
    #
    # Each unique row here represents one complete comment:
    # title + review_id
    # --------------------------------------------------------
    unique_reviews = (
        df[KEY_COLUMNS]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print("Unique reviews:", len(unique_reviews))

    # --------------------------------------------------------
    # Read AI labels.
    # The order of labels must match the order of unique_reviews.
    # --------------------------------------------------------
    unique_reviews["label_chatgpt_review"] = read_label_file(
        CHATGPT_LABELS_PATH,
        len(unique_reviews)
    )

    unique_reviews["label_perplexity_review"] = read_label_file(
        PERPLEXITY_LABELS_PATH,
        len(unique_reviews)
    )

    unique_reviews["label_copilot_review"] = read_label_file(
        COPILOT_LABELS_PATH,
        len(unique_reviews)
    )

    # --------------------------------------------------------
    # Remove old sentence-level AI labels and old final label.
    # label_m is kept unchanged.
    # --------------------------------------------------------
    old_columns_to_drop = [
        "label_perplexity",
        "label_copilot",
        "label",
        "label_chatgpt_review",
        "label_perplexity_review",
        "label_copilot_review",
    ]

    df = df.drop(
        columns=[column for column in old_columns_to_drop if column in df.columns]
    )

    # --------------------------------------------------------
    # Merge review-level AI labels back to sentence-level rows.
    #
    # Every sentence with the same title + review_id receives
    # the same AI labels.
    # --------------------------------------------------------
    merged_df = df.merge(
        unique_reviews,
        on=KEY_COLUMNS,
        how="left"
    )

    # --------------------------------------------------------
    # Safety check.
    # If this is not 0, something is wrong with the merge.
    # --------------------------------------------------------
    missing_count = merged_df["label_chatgpt_review"].isna().sum()

    print("Rows without assigned AI review labels:", missing_count)

    if missing_count > 0:
        print("\nWARNING: Some rows did not receive AI review labels.")
        print(
            merged_df[
                merged_df["label_chatgpt_review"].isna()
            ][KEY_COLUMNS + ["sentence_id", "text"]].head(20)
        )

    # --------------------------------------------------------
    # Create new final label using majority vote.
    # --------------------------------------------------------
    merged_df["label"] = merged_df.apply(majority_vote, axis=1)

    # --------------------------------------------------------
    # Save updated sentence-level dataset.
    # Original OPJDataset_clean.csv is not overwritten.
    # --------------------------------------------------------
    merged_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print("Saved updated dataset to:", OUTPUT_PATH)

    print("\nPreview:")
    print(
        merged_df[
            KEY_COLUMNS
            + [
                "sentence_id",
                "label_m",
                "label_chatgpt_review",
                "label_perplexity_review",
                "label_copilot_review",
                "label",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()
