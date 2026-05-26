import pandas as pd
import numpy as np
import re

# ============================================================
# PATHOVI
# ============================================================

CROFIREDA_PATH = "/home/mvlasic/OPJ-SinglePlayer/data_raw/crofireda_train.tsv"
CROFIREDA_OUT = "/home/mvlasic/OPJ-SinglePlayer/data_raw/crofireda_clean.tsv"

OPJ_PATH = "/home/mvlasic/OPJ-SinglePlayer/data_raw/OPJDataset.csv"
OPJ_OUT = "/home/mvlasic/OPJ-SinglePlayer/data_raw/OPJDataset_clean.csv"


# ============================================================
# FUNKCIJA ZA ČIŠĆENJE TEKSTA
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return None

    text = str(text).strip()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)

    if text == "":
        return None

    return text


# ============================================================
# 1. CRO-FIREDA PREPROCESSING
# ============================================================

cro = pd.read_csv(CROFIREDA_PATH, sep="\t", engine="python")
cro.columns = cro.columns.str.strip().str.lower()

cro["text"] = cro["text"].apply(clean_text)
cro["label"] = pd.to_numeric(cro["label"], errors="coerce")
cro = cro[cro["label"].isin([0, 1, 2])]

cro = cro.dropna(subset=["text", "label"])
cro["label"] = cro["label"].astype(int)

print("Cro-FiReDa size:", len(cro))
print("Cro-FiReDa label distribution:")
print(cro["label"].value_counts().sort_index())

cro.to_csv(CROFIREDA_OUT, sep="\t", index=False)

print("Cro-FiReDa saved:", CROFIREDA_OUT)


# ============================================================
# 2. OPJ DATASET PREPROCESSING
# ============================================================

opj = pd.read_csv(OPJ_PATH, sep=";", engine="python")
opj.columns = opj.columns.str.strip()

def final_label(row):
    vals = [
        row["label_m"],
        row["label_perplexity"],
        row["label_copilot"]
    ]

    vals = [v for v in vals if pd.notna(v)]

    if len(vals) == 0:
        return np.nan

    counts = pd.Series(vals).value_counts()

    if counts.iloc[0] >= 2:
        return int(counts.index[0])

    return int(np.median(vals))


opj["label"] = opj.apply(final_label, axis=1)
opj["text"] = opj["text"].apply(clean_text)

opj = opj.dropna(subset=["text", "label"])
opj["label"] = opj["label"].astype(int)

print("\nOPJDataset size:", len(opj))
print("OPJDataset label distribution:")
print(opj["label"].value_counts().sort_index())

opj.to_csv(OPJ_OUT, index=False, encoding="utf-8-sig")

print("OPJDataset saved:", OPJ_OUT)

print("\nPREPROCESSING DONE")