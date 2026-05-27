import pandas as pd
from pathlib import Path

# ============================================================
# PURPOSE OF THIS SCRIPT
# ============================================================
# This script updates the clean sentence-level dataset.
#
# Original situation:
# - OPJDataset_clean.csv contains one row per sentence.
# - It already has:
#     label_m            -> manual label, created by the student
#     label_perplexity   -> old Perplexity label, assigned per sentence
#     label_copilot      -> old Copilot label, assigned per sentence
#     label              -> old final/majority label
#
# Problem:
# - Perplexity, Copilot and ChatGPT should evaluate the sentiment
#   of the entire review/comment, not each sentence separately.
#
# Goal:
# - Keep the manual label_m exactly as it is.
# - Remove old sentence-level AI labels:
#     label_perplexity
#     label_copilot
#     label
# - Add new review-level AI labels:
#     label_chatgpt_review
#     label_perplexity_review
#     label_copilot_review
# - Assign the same review-level AI label to every sentence
#   that belongs to the same review_id.
# - Create a new final label using majority voting from:
#     label_m
#     label_chatgpt_review
#     label_perplexity_review
#     label_copilot_review
#
# The original OPJDataset_clean.csv is NOT overwritten.
# A new clean file is created:
#     OPJDataset_clean_review_labels.csv


# ============================================================
# INPUT FILES
# ============================================================
# CLEAN_DATASET_PATH:
# Sentence-level dataset.
# One row = one sentence.
# This file contains review_id and sentence_id.
CLEAN_DATASET_PATH = Path("data_raw/OPJDataset_clean.csv")

# REVIEW_LEVEL_PATH:
# Review-level dataset.
# One row = one whole review/comment.
# This file should already be created by grouping sentences
# with the same review_id.
REVIEW_LEVEL_PATH = Path("data_raw/OPJDataset_review_level.csv")

# These three text files contain AI labels for whole reviews.
# Each file must contain one label per line.
# The order of labels must match the order of rows in
# OPJDataset_review_level.csv.
CHATGPT_LABELS_PATH = Path("data_raw/review_labels_chatgpt.txt")
PERPLEXITY_LABELS_PATH = Path("data_raw/review_labels_perplexity.txt")
COPILOT_LABELS_PATH = Path("data_raw/review_labels_copilot.txt")


# ============================================================
# OUTPUT FILES
# ============================================================
# Review-level dataset with new AI review labels.
OUTPUT_REVIEW_PATH = Path("data_raw/OPJDataset_review_level_with_ai_labels.csv")

# New sentence-level clean dataset.
# This keeps label_m, removes old AI labels,
# and adds the new review-level AI labels.
OUTPUT_CLEAN_PATH = Path("data_raw/OPJDataset_clean_review_labels.csv")


# ============================================================
# LABEL MAPPING
# ============================================================
# Sentiment labels are standardized as:
# 0 = negative
# 1 = neutral
# 2 = positive
#
# The script accepts both Croatian and English label names,
# as well as already numeric labels.
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
    Convert one textual or numeric sentiment label into an integer.

    Accepted input examples:
    - "negativno", "negative", "0" -> 0
    - "neutralno", "neutral", "1" -> 1
    - "pozitivno", "positive", "2" -> 2

    If the label is not recognized, the script stops with an error.
    This prevents silently adding wrong labels into the dataset.
    """
    value = str(value).strip().lower()

    if value not in LABEL_MAP:
        raise ValueError(
            f"Nepoznata oznaka: '{value}'. "
            "Dozvoljeno: negativno, neutralno, pozitivno ili 0, 1, 2."
        )

    return LABEL_MAP[value]


def read_label_file(path, expected_count):
    """
    Read one AI label file and convert its labels to numbers.

    Parameters:
    - path: location of the .txt file with labels
    - expected_count: number of reviews in OPJDataset_review_level.csv

    The number of labels must be exactly the same as the number of reviews.
    If not, the labels cannot be safely matched to review_id values.
    """
    if not path.exists():
        raise FileNotFoundError(f"Nedostaje file: {path}")

    # Read all non-empty lines from the label file.
    with open(path, "r", encoding="utf-8") as file:
        raw_labels = [line.strip() for line in file.readlines() if line.strip()]

    # Convert labels from text to numbers.
    labels = [normalize_label(label) for label in raw_labels]

    # Safety check: one label must exist for every review.
    if len(labels) != expected_count:
        raise ValueError(
            f"File {path} ima {len(labels)} oznaka, "
            f"a review-level dataset ima {expected_count} reviewa. "
            "Broj oznaka mora biti isti."
        )

    return labels


def majority_vote(row):
    """
    Create the final label using majority voting.

    The final label is calculated from four sources:
    - label_m
    - label_chatgpt_review
    - label_perplexity_review
    - label_copilot_review

    If one sentiment class appears more often than the others,
    that class becomes the final label.

    Tie handling:
    - If there is a tie and neutral class 1 is one of the tied labels,
      the script chooses 1 because it is the safer middle category.
    - Otherwise, it returns the first most frequent label.
    """
    labels = [
        row["label_m"],
        row["label_chatgpt_review"],
        row["label_perplexity_review"],
        row["label_copilot_review"],
    ]

    # Remove missing labels and convert all values to integers.
    labels = [int(x) for x in labels if pd.notna(x)]

    if not labels:
        return None

    # Count how many times each label appears.
    counts = pd.Series(labels).value_counts()

    # Find the maximum vote count.
    top_count = counts.max()

    # Find all labels that have the maximum count.
    top_labels = counts[counts == top_count].index.tolist()

    # If there is a tie and neutral is part of it, choose neutral.
    if 1 in top_labels:
        return 1

    return int(top_labels[0])


def main():
    """
    Main workflow:
    1. Load sentence-level clean dataset.
    2. Load review-level dataset.
    3. Read new AI labels for whole reviews.
    4. Add AI labels to the review-level dataset.
    5. Remove old sentence-level AI labels from clean dataset.
    6. Merge the new review-level labels back to every sentence by review_id.
    7. Create a new final label by majority vote.
    8. Save new output files.
    """

    # --------------------------------------------------------
    # Check whether all required input datasets exist.
    # --------------------------------------------------------
    if not CLEAN_DATASET_PATH.exists():
        raise FileNotFoundError(f"Nedostaje clean dataset: {CLEAN_DATASET_PATH}")

    if not REVIEW_LEVEL_PATH.exists():
        raise FileNotFoundError(
            f"Nedostaje review-level dataset: {REVIEW_LEVEL_PATH}. "
            "Prvo pokreni scripts/create_review_level_dataset.py"
        )

    # --------------------------------------------------------
    # Load datasets.
    # --------------------------------------------------------
    clean_df = pd.read_csv(CLEAN_DATASET_PATH)
    review_df = pd.read_csv(REVIEW_LEVEL_PATH)

    # --------------------------------------------------------
    # Basic column checks.
    # review_id is necessary because it connects a whole review
    # with all sentence rows that belong to it.
    # --------------------------------------------------------
    if "review_id" not in clean_df.columns:
        raise ValueError("Clean dataset mora imati stupac 'review_id'.")

    if "review_id" not in review_df.columns:
        raise ValueError("Review-level dataset mora imati stupac 'review_id'.")

    if "label_m" not in clean_df.columns:
        raise ValueError("Clean dataset mora imati stupac 'label_m'.")

    review_count = len(review_df)

    print("Broj reviewa:", review_count)
    print("Broj rečenica u clean datasetu:", len(clean_df))

    # --------------------------------------------------------
    # Read new AI labels and add them to review-level dataset.
    # Each row in review_df represents one full review.
    # Therefore each AI label is assigned once per review.
    # --------------------------------------------------------
    review_df["label_chatgpt_review"] = read_label_file(
        CHATGPT_LABELS_PATH,
        review_count
    )

    review_df["label_perplexity_review"] = read_label_file(
        PERPLEXITY_LABELS_PATH,
        review_count
    )

    review_df["label_copilot_review"] = read_label_file(
        COPILOT_LABELS_PATH,
        review_count
    )

    # --------------------------------------------------------
    # Save review-level dataset with new AI labels.
    # This file is useful for checking labels per complete review.
    # --------------------------------------------------------
    review_df.to_csv(OUTPUT_REVIEW_PATH, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Remove old sentence-level AI labels.
    # These labels were created per sentence and should no longer
    # be used after switching to review-level AI annotation.
    #
    # label is also removed because it was based on the old labels.
    # A new label will be calculated below.
    # --------------------------------------------------------
    columns_to_drop = [
        "label_perplexity",
        "label_copilot",
        "label",
        "label_chatgpt_review",
        "label_perplexity_review",
        "label_copilot_review",
        "label_ai_review_majority",
    ]

    clean_df = clean_df.drop(
        columns=[col for col in columns_to_drop if col in clean_df.columns]
    )

    # --------------------------------------------------------
    # Keep only review_id and the new review-level labels.
    # Then merge them into the sentence-level dataset.
    #
    # Result:
    # Every sentence receives the same AI review label as all other
    # sentences with the same review_id.
    # --------------------------------------------------------
    labels_to_merge = review_df[
        [
            "review_id",
            "label_chatgpt_review",
            "label_perplexity_review",
            "label_copilot_review",
        ]
    ].drop_duplicates("review_id")

    clean_df = clean_df.merge(
        labels_to_merge,
        on="review_id",
        how="left"
    )

    # --------------------------------------------------------
    # Create new final label.
    # This is calculated from:
    # - manual sentence label_m
    # - ChatGPT review-level label
    # - Perplexity review-level label
    # - Copilot review-level label
    # --------------------------------------------------------
    clean_df["label"] = clean_df.apply(majority_vote, axis=1)

    # --------------------------------------------------------
    # Save the new clean dataset.
    # The original OPJDataset_clean.csv remains unchanged.
    # --------------------------------------------------------
    clean_df.to_csv(OUTPUT_CLEAN_PATH, index=False, encoding="utf-8-sig")

    # --------------------------------------------------------
    # Print summary for checking.
    # --------------------------------------------------------
    print("\nGotovo.")
    print("Nova review-level datoteka:", OUTPUT_REVIEW_PATH)
    print("Nova clean datoteka:", OUTPUT_CLEAN_PATH)

    print("\nStupci u novom clean datasetu:")
    print(clean_df.columns.tolist())

    print("\nPrvih 10 redova za provjeru:")
    print(
        clean_df[
            [
                "review_id",
                "sentence_id",
                "label_m",
                "label_chatgpt_review",
                "label_perplexity_review",
                "label_copilot_review",
                "label",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()