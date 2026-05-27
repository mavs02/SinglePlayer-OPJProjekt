import pandas as pd
from pathlib import Path

# ============================================================
# PURPOSE
# ============================================================
# This script extracts complete reviews from the original
# sentence-level clean dataset.
#
# Original dataset:
# - one row = one sentence
#
# Output dataset:
# - one row = one complete review/comment
#
# A unique review is identified by:
# title + review_id
#
# This is because review_id starts again from 1 when a new
# doctor/title begins.
#
# sentence_id is used only to preserve the correct sentence order
# inside each review.


INPUT_PATH = Path("data_raw/OPJDataset_clean.csv")
OUTPUT_CSV_PATH = Path("data_raw/reviews_for_ai_labeling.csv")
OUTPUT_TXT_PATH = Path("data_raw/reviews_for_ai_labeling.txt")

KEY_COLUMNS = ["title", "review_id"]


def majority_label(labels):
    """
    Creates a review-level manual label from sentence-level label_m.

    If most sentences in the review are positive, the review-level
    manual label becomes positive, etc.

    This is only for reference while labeling.
    """
    labels = pd.to_numeric(labels, errors="coerce").dropna().astype(int)

    if len(labels) == 0:
        return None

    counts = labels.value_counts()
    top_count = counts.max()
    top_labels = counts[counts == top_count].index.tolist()

    # If there is a tie and neutral is one of the tied labels,
    # choose neutral as the safer middle category.
    if 1 in top_labels:
        return 1

    return int(top_labels[0])


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required_columns = KEY_COLUMNS + ["sentence_id", "text", "label_m"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Clean text.
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]

    # Sort so sentences inside the same review stay in order.
    df = df.sort_values(KEY_COLUMNS + ["sentence_id"])

    # Create one row per complete review.
    reviews = (
        df.groupby(KEY_COLUMNS, as_index=False)
        .agg(
            full_review=("text", lambda sentences: " ".join(sentences)),
            number_of_sentences=("sentence_id", "count"),
            label_m_review=("label_m", majority_label),
        )
    )

    # Add review order so AI labels can be entered in the same order.
    reviews.insert(0, "review_order", range(1, len(reviews) + 1))

    # Save CSV for Excel/Sheets.
    reviews.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    # Save TXT for easier copy-paste into ChatGPT/Perplexity/Copilot.
    with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f:
        for _, row in reviews.iterrows():
            f.write(f"REVIEW_ORDER: {row['review_order']}\n")
            f.write(f"TITLE: {row['title']}\n")
            f.write(f"REVIEW_ID: {row['review_id']}\n")
            f.write(f"NUMBER_OF_SENTENCES: {row['number_of_sentences']}\n")
            f.write("FULL_REVIEW:\n")
            f.write(str(row["full_review"]))
            f.write("\n")
            f.write("=" * 80)
            f.write("\n\n")

    print("Done.")
    print("Number of unique reviews:", len(reviews))
    print("CSV saved to:", OUTPUT_CSV_PATH)
    print("TXT saved to:", OUTPUT_TXT_PATH)

    print("\nPreview:")
    print(
        reviews[
            [
                "review_order",
                "title",
                "review_id",
                "number_of_sentences",
                "label_m_review",
                "full_review",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    main()
