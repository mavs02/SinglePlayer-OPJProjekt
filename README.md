# Croatian Review Sentiment Analysis

## Project Report

## 1. Introduction

This project focuses on sentiment analysis of Croatian reviews. The main goal was to build a complete NLP pipeline for Croatian sentiment classification, from dataset preparation and annotation to model training, evaluation, comparison and deployment.

The task is a three-class sentiment classification problem:

```text
0 = negative
1 = neutral
2 = positive
```

The project compares several approaches:

```text
Method 1: Machine Learning
Method 2: Shallow Deep Learning
Method 3: Transformer-based Model
Method 4: GenAI / Zero-shot Model
```

The project uses two main datasets:

```text
1. Cro-FiReDa dataset
2. My manually annotated OPJ / NajDoktor dataset
```

Cro-FiReDa is used as the main public Croatian sentiment dataset for training, validation and in-domain testing. My OPJ / NajDoktor annotated dataset is used as an additional out-of-domain evaluation dataset.

An important methodological decision in this project was to work with the custom dataset at two levels:

```text
1. Review-level annotation
2. Sentence-level evaluation
```

The annotation was performed at review level, because the sentiment of a full review is usually clearer than the sentiment of an isolated sentence. After review-level annotation, the labels were distributed to the sentence-level dataset using `review_id`.

The final project also includes a Gradio demo deployed on Hugging Face Spaces.

---

## 2. Problem Definition

The task is supervised multi-class sentiment classification.

Given a Croatian review or sentence:

```text
x = Croatian text
```

the model predicts one of three possible labels:

```text
y ∈ {0, 1, 2}
```

where:

```text
0 = negative sentiment
1 = neutral sentiment
2 = positive sentiment
```

The model learns a function:

```text
f(x) → y
```

The goal is to classify unseen Croatian review text as negative, neutral or positive.

Because the datasets are imbalanced, the evaluation does not rely only on accuracy. The project also reports precision, recall and F1-score. Weighted averages are used in the final model comparison when available.

---

# 3. Repository Structure

The repository is organized by methodology, data, results and utility scripts.

```text
SinglePlayer-OPJProjekt/
│
├── Method1/
│   └── ML/
│
├── Method2/
│   └── DL/
│
├── Method3/
│   └── Transformer/
│       └── BERTic/
│
├── Method4/
│   └── GENAI/
│       └── Gemma/
│
├── data/
│   └── processed/
│
├── data_raw/
│
├── predictions/
│
├── results/
│
├── scripts/
│
├── .gitignore
├── LICENSE
└── README.md
```

Each main folder has a specific role in the project.

---

## 3.1 Method1 / ML

The folder `Method1/ML` contains the traditional machine learning approach.

It includes scripts such as:

```text
Train.py
Eval.py
Eval_5.py
Preprocess.py
```

This method was used to create a strong baseline. Traditional machine learning is useful because it is simple, fast, interpretable and does not require a GPU.

The main model used in this method was:

```text
TF-IDF + Logistic Regression
```

An additional model was also tested:

```text
TF-IDF + Balanced MLPClassifier
```

### What was used for what?

```text
TF-IDF
```

TF-IDF was used to transform Croatian text into numerical features. Machine learning models cannot directly process raw text, so the text first had to be converted into vectors. TF-IDF gives more importance to words that are frequent in a specific document but not too frequent across all documents.

```text
Logistic Regression
```

Logistic Regression was used as the main classical machine learning classifier. It is a standard and reliable baseline for text classification tasks.

```text
Balanced MLPClassifier
```

The MLPClassifier was used as an additional neural baseline on top of TF-IDF features. It was balanced to reduce the negative effect of class imbalance.

```text
5-fold cross-validation
```

Cross-validation was used to estimate how stable the machine learning model is during training. Instead of relying on only one split, the training data is split into several folds and the model is evaluated several times.

---

## 3.2 Method2 / DL

The folder `Method2/DL` contains the shallow deep learning approach.

It includes scripts such as:

```text
Train.py
Eval.py
eval_2.py
```

This method was used to test whether a neural network can learn useful sentiment patterns from Croatian review text.

The main model was:

```text
CNN / Improved CNN
```

### What was used for what?

```text
CNN
```

CNN stands for Convolutional Neural Network. In text classification, CNNs can detect local patterns such as short phrases, word combinations or sentiment-bearing expressions.

For example, phrases such as:

```text
"jako dobro"
"nije dobro"
"vrlo ljubazno"
"dugo čekanje"
```

can be useful for sentiment classification.

```text
Improved CNN
```

The improved CNN was introduced to improve the first deep learning result. It was used to handle the dataset better and improve performance, especially on the OPJ / NajDoktor annotated dataset.

CNN models are stronger than basic machine learning in some cases because they can learn patterns automatically. However, they are still weaker than transformer models when the meaning depends on broader context.

---

## 3.3 Method3 / Transformer / BERTic

The folder `Method3/Transformer/BERTic` contains the transformer-based approach.

The model used was:

```text
classla/bcms-bertic
```

BERTić is a transformer model trained for Bosnian, Croatian, Montenegrin and Serbian language varieties. It was used because it is more suitable for Croatian than general English-language models.

### What was used for what?

```text
BERTić
```

BERTić was used as the main transformer model for Croatian sentiment classification. It can understand context better than TF-IDF and CNN models.

For example, BERTić can better interpret cases where sentiment is not based only on one word, but on the meaning of the whole sentence.

```text
Hugging Face Transformers
```

The Hugging Face Transformers library was used to load the pretrained BERTić model, tokenize the text, fine-tune the model and run evaluation.

```text
Tokenizer
```

The tokenizer converts raw Croatian text into token IDs that the transformer model can process.

```text
Fine-tuning
```

Fine-tuning means that the pretrained BERTić model was further trained on the sentiment classification task. Instead of training a language model from zero, the project adapts an already pretrained model to the specific task.

The transformer model was evaluated on:

```text
Test 1: Cro-FiReDa test set
Test 2: OPJ / NajDoktor annotated dataset
```

BERTić was the best-performing model in the project.

---

3.4 Method4 / GENAI / Gemma

The folder Method4/GENAI/Gemma contains the generative AI prompt-based approach for Croatian sentiment classification.

This method includes two prompting strategies:

1. Zero-shot classification
2. One-shot classification

The models used were:

google/gemma-3-4b-it
google/gemma-3-1b-it

The zero-shot experiment uses google/gemma-3-4b-it, while the one-shot experiment uses google/gemma-3-1b-it.

This method was used to test whether a generative model can classify Croatian sentiment without supervised fine-tuning. Unlike the ML, DL and Transformer methods, Gemma was not trained on the Cro-FiReDa training set. Instead, the model received a prompt and returned a sentiment label.

The relevant files are:

Method4/GENAI/Gemma/run_gemma_zero_shot.py
Method4/GENAI/Gemma/genai_one_shot.py
What was used for what?
Gemma3

Gemma3 was used as a prompt-based sentiment classifier. The model classified Croatian reviews by receiving an instruction prompt and generating one of the allowed labels:

0 = negative
1 = neutral
2 = positive
Zero-shot classification

Zero-shot classification was used to test whether a generative AI model can classify Croatian sentiment directly from instructions, without receiving any labelled example in the prompt.

In the zero-shot script, each Croatian sentence is inserted into a prompt that explains the task and asks the model to return only one number: 0, 1 or 2.

The zero-shot script evaluates the model on both test sets:

Test 1: Cro-FiReDa
Test 2: OPJ / NajDoktor annotated dataset

The script saves results and predictions into:

results/Trial_4_Gemma_zero_shot_crofireda.txt
results/Trial_4_Gemma_zero_shot_golden.txt
results/Trial_4_Gemma_zero_shot_summary.txt

predictions/Prediction_Gemma_zero_shot_crofireda.csv
predictions/Prediction_Gemma_zero_shot_golden.csv

Note: although the filename contains golden, in the report this dataset is described as the OPJ / NajDoktor annotated dataset.

One-shot classification

One-shot classification was used to test whether the generative model performs better when it receives one labelled example before classifying the target review.

In the one-shot script, the prompt contains:

1. task instruction
2. label explanation
3. one labelled example review
4. target review for classification

The prompt structure is:

Task: classify the sentiment of a Croatian review.
Labels:
0 = negative
1 = neutral
2 = positive

Example:
Review: "..."
Label: ...

Now classify the following review.
Review: "..."
Label:

This helps the model understand the expected label format and how the task should be solved.

The one-shot script uses:

data_raw/OPJDataset_clean_review_labels.csv

and saves outputs into:

results/method4_genai_one_shot_results.csv
results/method4_genai_one_shot_metrics.txt

The script also automatically selects one example from the dataset. If no manual index is selected, it preferably selects one neutral example, because neutral sentiment is often the most ambiguous class.

Why this method was included

The GenAI / Gemma method was included as an additional comparison with supervised models.

It helps answer the following question:

Can a generative AI model classify Croatian sentiment using only prompting, without supervised fine-tuning?

The results showed that Gemma can be useful as a prompt-based baseline, especially because it does not require training. However, it was less reliable than the supervised BERTić model.

Difference between zero-shot and one-shot

The difference is:

Zero-shot:
The model receives only the instruction and the review text.

One-shot:
The model receives the instruction, one labelled example and then the review text.

Zero-shot tests whether the model understands the task from instructions only.

One-shot tests whether one example helps the model understand the expected classification format and label meaning.

Limitations of the GenAI / Gemma approach

The GenAI approach is flexible, but it is less stable than supervised fine-tuning.

Typical limitations include:

- predictions depend on prompt wording
- the model can return invalid output if the answer is not cleaned
- neutral reviews are difficult to classify consistently
- polite negative reviews can be misclassified as positive
- one example may help formatting but does not replace supervised training
- the model was not fine-tuned specifically for this dataset

Overall, Gemma was used as a prompt-based comparison method, while BERTić remained the strongest supervised model in the project.

# 4. Dataset

## 4.1 Dataset Overview

The project uses two main datasets:

```text
1. Cro-FiReDa
2. OPJ / NajDoktor annotated dataset
```

Cro-FiReDa is used for model training, validation and in-domain testing. The OPJ / NajDoktor annotated dataset is used as an additional out-of-domain test set.

---

## 4.2 Cro-FiReDa Dataset

Cro-FiReDa is the main public Croatian sentiment dataset used in this project.

The processed files are stored in:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

The split contains:

```text
Train: 5144 examples
Validation: 343 examples
Test: 1372 examples
```

Label distribution:

```text
Train:
positive = 1096
negative = 697
neutral = 3351
total = 5144

Validation:
positive = 73
negative = 46
neutral = 224
total = 343

Test:
positive = 293
negative = 186
neutral = 893
total = 1372
```

Cro-FiReDa is used as:

```text
Test 1 = in-domain test set
```

This means that the model is trained and tested on data from the same general source/distribution.

---

## 4.3 OPJ / NajDoktor Annotated Dataset

The OPJ / NajDoktor annotated dataset is my own manually annotated dataset created for this project.

It was collected from Croatian reviews from the NajDoktor website. This source was selected because it contains real Croatian user-generated text, including praise, criticism, recommendations, neutral descriptions and mixed opinions.

The dataset was prepared at two levels:

```text
1. Review-level dataset
2. Sentence-level dataset
```

The review-level dataset was used for annotation. The sentence-level dataset was used for final evaluation.

Important files:

```text
data_raw/OPJDataset_clean.csv
data_raw/OPJDataset_clean_review_labels.csv
data_raw/OPJDataset_review_agreement_labels.csv
data_raw/OPJDataset_clean_review_labels_final.csv
```

The final annotated dataset used in evaluation is:

```text
data_raw/OPJDataset_clean_review_labels_final.csv
```

It contains:

```text
Review rows: 139
Sentence rows: 536
```

Final label distribution:

```text
0 = negative: 54
1 = neutral: 78
2 = positive: 404
Total: 536
```

This dataset is used as:

```text
Test 2 = out-of-domain annotated dataset
```

This means that the models trained on Cro-FiReDa are tested on a different source and domain. This is important because it measures generalization.

---

# 5. Dataset Creation

The OPJ / NajDoktor dataset was created manually.

The workflow was:

```text
1. Collect reviews from NajDoktor.
2. Clean the raw data.
3. Preserve metadata such as URL, title, review ID and year.
4. Create a review-level dataset.
5. Export reviews for manual and AI-assisted annotation.
6. Annotate reviews manually.
7. Collect labels from AI annotators.
8. Calculate inter-rater agreement.
9. Create final label using majority vote.
10. Distribute review-level labels to sentence-level rows.
11. Use the final sentence-level dataset for evaluation.
```

The reason for review-level annotation is that sentiment is often clearer in the whole review than in one isolated sentence.

For example:

```text
"Pregled je trajao vrlo kratko."
```

This sentence can look neutral by itself. However, if the whole review is a complaint, then the sentence contributes to negative sentiment.

Because of this, the project first assigned sentiment to the full review and then distributed the review-level label to its sentences.

---

# 6. Annotation

The annotation process used:

```text
1. Manual annotation
2. AI-assisted annotation
```

Manual annotation was performed by the project author.

Since the project was completed independently and there was no second human annotator, AI tools were used as additional annotators. They were used to compare agreement with the manual labels, not to replace manual work.

The AI annotators included:

```text
ChatGPT
Perplexity
Copilot
```

The labels were:

```text
0 = negative
1 = neutral
2 = positive
```

Important annotation columns:

```text
label_m = manual review-level label
label_chatgpt_review = ChatGPT review-level label
label_perplexity_review = Perplexity review-level label
label_copilot_review = Copilot review-level label
label = final majority-vote label
```

The final label was created using majority voting:

```text
Majority vote:
label_m + label_chatgpt_review + label_perplexity_review + label_copilot_review

Tie rule:
manual label wins
```

The manual label wins in ties because manual annotation was treated as the reference decision in unclear cases.

---

# 7. Inter-rater Agreement

Inter-rater agreement was calculated using Fleiss’ Kappa.

The agreement was calculated at review level because the annotation was review-based.

Dataset used for agreement:

```text
data_raw/OPJDataset_clean_review_labels.csv
```

Agreement output:

```text
data_raw/OPJDataset_review_agreement_labels.csv
results/fleiss_kappa.txt
results/fleiss_kappa_summary.csv
```

Number of samples:

```text
Review rows: 139
Sentence rows after label distribution: 536
```

Two agreement settings were calculated.

---

## 7.1 AI Chatbots Only

This comparison included only the AI annotators.

Result:

```text
Samples: 139
Fleiss’ Kappa: 0.3795
Interpretation: Fair agreement
```

This means that the AI annotators had fair agreement with each other. They usually agreed on clearly positive and clearly negative reviews, but disagreed more often on neutral or mixed reviews.

---

## 7.2 Manual Review Label + AI Chatbots

This comparison included the manual label and the AI labels.

Result:

```text
Samples: 139
Fleiss’ Kappa: 0.1251
Interpretation: Slight agreement
```

This result shows that agreement between the manual annotation and AI annotations was low.

The main reasons for disagreement were:

```text
- some reviews contained both praise and criticism
- some negative reviews were written politely
- factual medical descriptions were sometimes interpreted as neutral
- AI tools sometimes over-predicted positive sentiment
- some sentences needed the full review context
```

This confirms that sentiment annotation in Croatian medical reviews is difficult and context-dependent.

---

# 8. Pre-processing

Pre-processing was necessary to prepare the text for modelling.

The project used different pre-processing steps depending on the method.

## 8.1 General Pre-processing

General pre-processing included:

```text
- cleaning raw text
- removing empty or invalid rows
- preserving review metadata
- preparing train, validation and test files
- standardizing labels as 0, 1 and 2
- preparing CSV files for model training and evaluation
```

## 8.2 Pre-processing for Machine Learning

For machine learning models, the text was transformed using TF-IDF.

Used for:

```text
TF-IDF + Logistic Regression
TF-IDF + Balanced MLPClassifier
```

Why TF-IDF was used:

```text
- converts raw text into numerical vectors
- works well for classical machine learning
- captures important words and word combinations
- is simple and fast
- creates a strong baseline
```

## 8.3 Pre-processing for CNN

For CNN models, text had to be tokenized and converted into sequences.

Used for:

```text
CNN
Improved CNN
```

Why tokenization and sequence preparation were used:

```text
- neural networks need numerical input
- each word/token is converted into an index
- sequences are padded or truncated to the same length
- the model learns patterns from token sequences
```

## 8.4 Pre-processing for BERTić

For BERTić, the Hugging Face tokenizer was used.

Used for:

```text
classla/bcms-bertic
```

Why BERTić tokenizer was used:

```text
- transformer models require token IDs
- the tokenizer splits Croatian text into subword units
- it creates attention masks
- it prepares text in the exact format expected by BERTić
```

## 8.5 Pre-processing for Gemma

For Gemma, examples were prepared as prompts.

Used for:

```text
GenAI / zero-shot classification
```

Why prompting was used:

```text
- Gemma is a generative model
- it does not need supervised fine-tuning in this setup
- the model receives instructions and classifies sentiment from the prompt
```

---

# 9. Methodology

## 9.1 Method 1: Machine Learning

The first method was a classical machine learning baseline.

Models:

```text
TF-IDF + Logistic Regression
TF-IDF + Balanced MLPClassifier
```

The main purpose of this method was to create a baseline for comparison.

### TF-IDF + Logistic Regression

This model was used because it is a common and reliable baseline in text classification.

TF-IDF extracts important words and phrases from text. Logistic Regression then learns which features are associated with negative, neutral or positive sentiment.

Two values of the regularization parameter were tested:

```text
C = 1.0
C = 5.0
```

The model with `C = 5.0` performed better and was used in the final comparison.

### TF-IDF + Balanced MLPClassifier

This model was added as another machine learning baseline. It still uses TF-IDF features, but the classifier is a small neural network.

The balanced version was used because the datasets are imbalanced. Positive and neutral examples appear more often than negative examples.

---

## 9.2 Method 2: Shallow Deep Learning

The second method used a CNN model.

Models:

```text
CNN
Improved CNN
```

CNN was used because convolutional layers can detect local textual patterns. In sentiment analysis, short word combinations often carry sentiment.

Examples:

```text
"jako dobro"
"nije zadovoljna"
"dugo čekanje"
"vrlo stručan"
```

The improved CNN was introduced to improve the first CNN result. It performed better on the OPJ / NajDoktor annotated dataset than the initial CNN.

However, CNN still has limitations. It can capture local patterns, but it does not understand context as well as transformer models.

---

## 9.3 Method 3: Transformer Model

The third method used BERTić:

```text
classla/bcms-bertic
```

BERTić was used because it is a pretrained transformer model suitable for Croatian and related South Slavic languages.

This method uses:

```text
Hugging Face Transformers
PyTorch
BERTić tokenizer
BERTić sequence classification model
```

BERTić was fine-tuned on the sentiment classification task.

This method was expected to perform best because transformer models understand context better than TF-IDF and CNN approaches.

For example, BERTić can better handle:

```text
- indirect sentiment
- longer context
- mixed sentiment
- negation
- Croatian language morphology
- words that change meaning depending on context
```

BERTić was the best-performing model in the final evaluation.

---

## 9.4 Method 4: GenAI / Zero-shot

The fourth method used Gemma3 in a zero-shot setting.

This means the model was not trained on the training dataset. Instead, it received a prompt and predicted the sentiment label.

This method was included because generative AI models are increasingly used for text classification and annotation. The goal was to compare a prompt-based approach with supervised training.

Gemma3 was useful as a baseline, but it was weaker than fine-tuned BERTić.

---

# 10. Results and Discussion

## 10.1 Final Model Comparison

| #      | Method                | Algorithm                       | Train             | Test 1 Precision | Test 1 Recall | Test 1 F1 | Test 1 Accuracy | Test 2 Precision | Test 2 Recall | Test 2 F1 | Test 2 Accuracy |
| ------ | --------------------- | ------------------------------- | ----------------- | ---------------: | ------------: | --------: | --------------: | ---------------: | ------------: | --------: | --------------: |
| 1.a.i  | Machine learning      | TF-IDF + Logistic Regression    | Train / 5-fold CV |             0.72 |          0.74 |      0.72 |          0.7354 |             0.56 |          0.43 |      0.43 |          0.4254 |
| 1.a.ii | Machine learning      | TF-IDF + Balanced MLPClassifier | Train / 5-fold CV |             0.73 |          0.67 |      0.69 |          0.6742 |             0.54 |          0.43 |      0.45 |          0.4291 |
| 2.a    | Shallow Deep learning | CNN / Improved                  | TRAIN             |             0.67 |          0.70 |      0.65 |          0.6990 |             0.56 |          0.60 |      0.54 |          0.6007 |
| 3.a    | Transformers          | BERTić                          | TRAIN             |             0.84 |          0.84 |      0.84 |          0.8360 |             0.72 |          0.74 |      0.73 |          0.7351 |
| 3.b    | GenAI / Zero-shot     | GENAI Gemma3                    | Zero-shot         |             0.79 |          0.73 |      0.73 |          0.7252 |             0.58 |          0.51 |      0.54 |          0.5093 |

---

## 10.2 Best Model

The best-performing model on Test 1, Cro-FiReDa, was:

```text
BERTić
```

Result:

```text
Precision = 0.84
Recall = 0.84
F1 = 0.84
Accuracy = 0.8360
```

The best-performing model on Test 2, OPJ / NajDoktor annotated dataset, was also:

```text
BERTić
```

Result:

```text
Precision = 0.72
Recall = 0.74
F1 = 0.73
Accuracy = 0.7351
```

This shows that the transformer-based model generalized better than the machine learning, CNN and zero-shot GenAI approaches.

---

## 10.3 Interpretation of Results

The results show that transformer-based modelling achieved the strongest overall performance.

BERTić performed best because it can model context. This is important for Croatian reviews, where sentiment is often not expressed with one simple positive or negative word.

The TF-IDF + Logistic Regression model performed reasonably well on Cro-FiReDa, especially after hyperparameter adjustment. However, its performance dropped on the OPJ / NajDoktor annotated dataset. This shows that traditional machine learning has weaker out-of-domain generalization.

The improved CNN performed better than the first CNN on the annotated dataset, but it was still weaker than BERTić. CNN can capture local patterns, but it does not understand full sentence or review context as deeply as a transformer.

The Gemma3 zero-shot model performed reasonably well without supervised training, especially on Cro-FiReDa. However, it was less reliable on the OPJ / NajDoktor annotated dataset. This suggests that zero-shot prompting can be useful as a baseline, but supervised fine-tuning is better for this task.

---

# 11. Confusion Matrix Code

The following code can be used to generate a confusion matrix for any prediction file.

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Change this path depending on the prediction file
PREDICTIONS_PATH = "predictions/Prediction_4_BERTic_annotated_test.csv"

df = pd.read_csv(PREDICTIONS_PATH)

# Adjust column names if needed
y_true = df["true_label"]
y_pred = df["predicted_label"]

labels = [0, 1, 2]
display_labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(y_true, y_pred, labels=labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(values_format="d")
plt.title("Confusion Matrix")
plt.show()
```

If the prediction file has a different name or different column names, the path and column names should be adjusted.

---

# 12. Error Analysis

## 12.1 Machine Learning Error Analysis

The best classical machine learning method was:

```text
TF-IDF + Logistic Regression
```

This model performed well on Cro-FiReDa, but its performance dropped on the OPJ / NajDoktor annotated dataset.

The main limitation is that TF-IDF does not understand deeper context. It relies mostly on words and short phrases.

Typical errors:

```text
- polite negative reviews
- mixed reviews with both praise and criticism
- neutral factual sentences inside negative reviews
- medical descriptions that do not contain clear sentiment words
- sentences where the meaning depends on the full review
```

Example:

```text
"Pregled je trajao četiri minute."
```

This sentence may look neutral, but in the full review it can express dissatisfaction.

A TF-IDF model may miss this because there is no strongly negative word.

---

## 12.2 Deep Learning Error Analysis

The improved CNN performed better than the basic CNN, especially on the OPJ / NajDoktor annotated dataset.

However, CNN still has problems with broader context.

Typical errors:

```text
- over-predicting frequent classes
- missing weak negative sentiment
- confusing neutral medical information with sentiment
- failing when one sentence depends on the full review context
```

This is important because the project uses review-level annotation but sentence-level evaluation. Some sentence-level examples inherit the sentiment of the full review even if the sentence itself does not contain explicit sentiment.

---

## 12.3 Transformer Error Analysis

BERTić was the best model, but it still made mistakes.

Typical errors:

```text
- mixed sentiment reviews
- neutral-positive borderline examples
- polite but negative comments
- factual medical sentences with review-level sentiment
- examples where full review context is necessary
```

BERTić performs better because it understands context, but the task is still difficult because sentiment annotation can be subjective.

---

## 12.4 GenAI / Gemma Error Analysis

Gemma3 was used as a zero-shot model.

Typical errors:

```text
- inconsistent classification of neutral reviews
- over-interpreting polite language as positive
- missing negative meaning in indirect criticism
- depending strongly on prompt wording
```

The zero-shot approach is useful because it does not require training, but it is less stable than a fine-tuned supervised model.

---

# 13. Difficult and Interesting Cases

The most difficult cases are those where sentiment is not explicit.

Example:

```text
"Doktorica je bila korektna, ali nisam dobila konkretno rješenje."
```

This is difficult because it contains both positive and negative elements.

Example:

```text
"Čekala sam dugo na pregled."
```

This may be neutral as a factual statement, but negative if the whole review is a complaint.

Example:

```text
"Sve je bilo u redu."
```

This can be weakly positive or neutral, depending on the annotation guidelines.

These examples show why review-level annotation was important. A single sentence does not always contain enough information to determine sentiment accurately.

---

# 14. Demo

A demo application was created using Gradio and deployed on Hugging Face Spaces.

The demo allows the user to enter a Croatian review text and receive a sentiment prediction:

```text
0 = negative
1 = neutral
2 = positive
```

Demo link:

```text
https://huggingface.co/spaces/mavs02/croatian-review-sentiment-analysis
```

The demo is available as a Hugging Face Space under the name:

```text
mavs02/croatian-review-sentiment-analysis
```

### What was Gradio used for?

Gradio was used to create a simple web interface for the sentiment analysis model.

Instead of running the model only from the command line, the user can enter a review in a web form and receive a prediction.

### What was Hugging Face Spaces used for?

Hugging Face Spaces was used to publicly host the demo application.

This makes the project accessible online and allows other users to test the model without installing the code locally.

---

# 15. Technologies and Libraries Used

## Python

Python was used as the main programming language for the entire project.

Used for:

```text
- data processing
- dataset preparation
- model training
- model evaluation
- metric calculation
- result export
- demo development
```

## pandas

pandas was used for working with CSV files and tabular data.

Used for:

```text
- reading datasets
- cleaning data
- merging review-level and sentence-level labels
- exporting final CSV files
- preparing result tables
```

## NumPy

NumPy was used for numerical operations.

Used for:

```text
- array handling
- metric preparation
- model input preparation
```

## scikit-learn

scikit-learn was used for classical machine learning and evaluation.

Used for:

```text
- TF-IDF vectorization
- Logistic Regression
- MLPClassifier
- train/test utilities
- precision, recall, F1-score and accuracy
- confusion matrix
- cross-validation
```

## PyTorch

PyTorch was used for deep learning and transformer-based modelling.

Used for:

```text
- CNN training
- neural network operations
- BERTić fine-tuning
- GPU-based training
```

## Hugging Face Transformers

Hugging Face Transformers was used for the transformer model.

Used for:

```text
- loading BERTić
- tokenizing text
- fine-tuning sequence classification model
- running transformer evaluation
```

## Gradio

Gradio was used for the demo interface.

Used for:

```text
- creating a simple web application
- allowing users to input Croatian review text
- displaying predicted sentiment
```

## Hugging Face Spaces

Hugging Face Spaces was used for deployment.

Used for:

```text
- hosting the Gradio demo
- making the model accessible online
- providing a public demo link
```

## AI Tools

AI tools were used as additional annotators.

Used for:

```text
- review-level sentiment annotation
- comparison with manual labels
- inter-rater agreement calculation
```

AI annotators included:

```text
ChatGPT
Perplexity
Copilot
```

---

# 16. How to Run

Clone the repository:

```bash
git clone https://github.com/mavs02/SinglePlayer-OPJProjekt.git
cd SinglePlayer-OPJProjekt
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers datasets matplotlib gradio
```

Run machine learning training:

```bash
python Method1/ML/Train.py
```

Run machine learning evaluation:

```bash
python Method1/ML/Eval.py
```

Run deep learning training:

```bash
python Method2/DL/Train.py
```

Run deep learning evaluation:

```bash
python Method2/DL/Eval.py
```

Run BERTić training:

```bash
python Method3/Transformer/BERTic/Train.py
```

Run BERTić evaluation:

```bash
python Method3/Transformer/BERTic/Eval.py
```

Run inter-rater agreement calculation:

```bash
python scripts/fleiss_kappa.py
```

Run final report generation:

```bash
python results/create_final_summary_from_txt.py
```

---

# 17. Links

## Code

```text
https://github.com/mavs02/SinglePlayer-OPJProjekt
```

## Dataset Files

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data_raw/OPJDataset_clean_review_labels_final.csv
```

## Results

```text
results/final_report.md
results/final_model_comparison.csv
results/fleiss_kappa.txt
results/fleiss_kappa_summary.csv
```

## Demo

```text
https://huggingface.co/spaces/mavs02/croatian-review-sentiment-analysis
```

## Model / Demo Hosting

```text
https://huggingface.co/spaces/mavs02/croatian-review-sentiment-analysis
```

---

# 18. Conclusion

This project implemented a full Croatian sentiment analysis pipeline.

The project included:

```text
- dataset preparation
- review-level annotation
- AI-assisted annotation
- inter-rater agreement calculation
- sentence-level dataset creation
- machine learning baseline
- shallow deep learning model
- transformer-based model
- GenAI zero-shot model
- model comparison
- Gradio demo deployment
```

The most important methodological decision was to annotate the custom dataset at review level and then distribute the labels to sentence-level rows. This was done because the sentiment of medical reviews is often easier to determine from the complete review than from isolated sentences.

The best-performing model was BERTić. It achieved the strongest results on both Cro-FiReDa and the OPJ / NajDoktor annotated dataset.

The results show that transformer-based models are the most effective approach for Croatian sentiment classification in this project. Traditional machine learning, CNN and zero-shot GenAI methods are useful baselines, but BERTić gives the best overall performance.

As a final step, the project was deployed as a Gradio demo on Hugging Face Spaces, which makes the model accessible through a simple web interface.

---

# 19. Drawbacks and Limitations

The project has several limitations:

```text
1. The OPJ / NajDoktor annotated dataset is relatively small.
2. The dataset is imbalanced, with positive reviews as the dominant class.
3. Manual annotation was done by one person.
4. AI annotators were used as additional annotators, but they cannot fully replace human annotators.
5. Annotation was performed at review level, while final evaluation was sentence-level.
6. Some sentence-level examples do not contain explicit sentiment by themselves.
7. Medical reviews contain factual, polite and context-dependent language.
8. Some models struggle with minority classes.
9. Zero-shot GenAI results depend strongly on prompt formulation.
10. Domain shift between Cro-FiReDa and NajDoktor affects performance.
```

---

# 20. Future Work

Future improvements could include:

```text
1. Expanding the OPJ / NajDoktor annotated dataset.
2. Adding another human annotator.
3. Creating more detailed annotation guidelines.
4. Separating review-level and sentence-level classification experiments.
5. Testing more Croatian and multilingual transformer models.
6. Improving the Gradio interface.
7. Adding confidence scores to the demo.
8. Performing more detailed manual error analysis.
9. Applying data augmentation or class balancing.
10. Testing the model on other Croatian review domains.
```

---

# 21. References

```text
Cro-FiReDa dataset
NajDoktor review source
BERTić: classla/bcms-bertic
Hugging Face Transformers documentation
Scikit-learn documentation
PyTorch documentation
Gradio documentation
Project repository: https://github.com/mavs02/SinglePlayer-OPJProjekt
Hugging Face Space: https://huggingface.co/spaces/mavs02/croatian-review-sentiment-analysis
```

---

# 22. License

This project is released under the CC0-1.0 license.
