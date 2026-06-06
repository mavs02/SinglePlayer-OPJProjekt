# Method 4: GENAI / Gemma

This folder contains an additional GENAI experiment for Croatian sentiment classification.

## Model

- `google/gemma-3-4b-it`

## Dataset

This experiment uses the existing project test dataset:

- `data/processed/test.csv`

No additional dataset was created for this experiment.

## Label scheme

- 0 = negative
- 1 = neutral
- 2 = positive

## Method

Gemma was used in a zero-shot setting. The model was not fine-tuned on the project dataset. Each Croatian review sentence was inserted into a prompt and the model was instructed to return only one sentiment label: 0, 1 or 2.

This experiment is used as an additional comparison with the supervised models implemented in the project:

- Method 1: Machine Learning
- Method 2: Deep Learning
- Method 3: Transformer / BERTić
- Method 4: GENAI / Gemma

## Output files

The script saves the results into the shared project folders:

- `results/Trial_4_Gemma_zero_shot.txt`
- `predictions/Prediction_Gemma_zero_shot.csv`

## Note

This is an additional GENAI experiment and does not replace the required ML, DL and Transformer models.
MD