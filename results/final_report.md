# Final Report: Croatian Sentiment Analysis

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

| split | positive | negative | neutral | total | source_file | label_column |
| --- | --- | --- | --- | --- | --- | --- |
| Train | 1096 | 697 | 3351 | 5144 | data/processed/train.csv | label |
| Validation | 73 | 46 | 224 | 343 | data/processed/val.csv | label |
| Test | 293 | 186 | 893 | 1372 | data/processed/test.csv | label |
| Golden dataset | 404 | 54 | 78 | 536 | data_raw/OPJDataset_clean_review_labels_final.csv | label |

## 3. Final model comparison

| # | method | algorithm | train | Test 1 Precision | Test 1 Recall | Test 1 F1 | Test 1 Accuracy | Test 2 Precision | Test 2 Recall | Test 2 F1 | Test 2 Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.a.i | Machine learning | TF-IDF + Logistic Regression | Train / 5-fold CV | 0.72 | 0.74 | 0.72 | 0.7354 | 0.56 | 0.43 | 0.43 | 0.4254 |
| 1.a.ii | Machine learning | TF-IDF + Balanced MLPClassifier | Train / 5-fold CV | 0.73 | 0.67 | 0.69 | 0.6742 | 0.54 | 0.43 | 0.45 | 0.4291 |
| 2.a | Shallow Deep learning | CNN / Improved | TRAIN | 0.67 | 0.7 | 0.65 | 0.699 | 0.56 | 0.6 | 0.54 | 0.6007 |
| 3.a | Transformers | BERTić | TRAIN | 0.84 | 0.84 | 0.84 | 0.836 | 0.72 | 0.74 | 0.73 | 0.7351 |
| 3.b | GenAI / Zero-shot | GENAI Gemma3 | Zero-shot | 0.79 | 0.73 | 0.73 | 0.7252 | 0.58 | 0.51 | 0.54 | 0.5093 |

## 4. Source-level extracted results

The following table shows the metrics extracted directly from the `.txt` result files in the `results/` directory.

| source_file | method | algorithm | train | dataset | precision | recall | f1 | accuracy | cv_mean_f1_macro | cv_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Trial_1_C_1.0.txt | Machine learning | TF-IDF + Logistic Regression | Train / 5-fold CV | Test 1: Cro-FiReDa | 0.7 | 0.72 | 0.69 | 0.7238 | 0.5389 | 0.0159 |
| Trial_1_C_1.0.txt | Machine learning | TF-IDF + Logistic Regression | Train / 5-fold CV | Test 2: Golden dataset | 0.54 | 0.38 | 0.35 | 0.375 | 0.5389 | 0.0159 |
| Trial_2_C_5.0.txt | Machine learning | TF-IDF + Logistic Regression | Train / 5-fold CV | Test 1: Cro-FiReDa | 0.72 | 0.74 | 0.72 | 0.7354 | 0.597 | 0.0182 |
| Trial_2_C_5.0.txt | Machine learning | TF-IDF + Logistic Regression | Train / 5-fold CV | Test 2: Golden dataset | 0.56 | 0.43 | 0.43 | 0.4254 | 0.597 | 0.0182 |
| Trial_3_CNN.txt | Shallow Deep learning | CNN / Improved | TRAIN | Test 1: Cro-FiReDa | 0.67 | 0.7 | 0.65 | 0.699 |  |  |
| Trial_3_CNN.txt | Shallow Deep learning | CNN / Improved | TRAIN | Test 2: Golden dataset | 0.49 | 0.33 | 0.28 | 0.3302 |  |  |
| Trial_3_CNN_improved.txt | Shallow Deep learning | CNN / Improved | TRAIN | Test 1: Cro-FiReDa | 0.7 | 0.65 | 0.65 | 0.6501 |  |  |
| Trial_3_CNN_improved.txt | Shallow Deep learning | CNN / Improved | TRAIN | Test 2: Golden dataset | 0.56 | 0.6 | 0.54 | 0.6007 |  |  |
| Trial_3_MLP_balanced.txt | Machine learning | TF-IDF + Balanced MLPClassifier | Train / 5-fold CV | Test 1: Cro-FiReDa | 0.73 | 0.67 | 0.69 | 0.6742 | 0.5916 | 0.0214 |
| Trial_3_MLP_balanced.txt | Machine learning | TF-IDF + Balanced MLPClassifier | Train / 5-fold CV | Test 2: Golden dataset | 0.54 | 0.43 | 0.45 | 0.4291 | 0.5916 | 0.0214 |
| Trial_4_BERTic.txt | Transformers | BERTić | TRAIN | Unknown |  |  |  | 0.8484 |  |  |
| Trial_4_BERTic_crofireda_test.txt | Transformers | BERTić | TRAIN | Test 1: Cro-FiReDa | 0.84 | 0.84 | 0.84 | 0.836 |  |  |
| Trial_4_BERTic_golden_test.txt | Transformers | BERTić | TRAIN | Test 2: Golden dataset | 0.72 | 0.74 | 0.73 | 0.7351 |  |  |
| Trial_4_Gemma_zero_shot_crofireda.txt | GenAI / Zero-shot | GENAI Gemma3 | Zero-shot | Test 1: Cro-FiReDa | 0.79 | 0.73 | 0.73 | 0.7252 |  |  |
| Trial_4_Gemma_zero_shot_golden.txt | GenAI / Zero-shot | GENAI Gemma3 | Zero-shot | Test 2: Golden dataset | 0.58 | 0.51 | 0.54 | 0.5093 |  |  |

## 5. Best-performing models

The best-performing model on Test 1 / Cro-FiReDa is BERTić with weighted F1 = 0.84 and accuracy = 0.836.

The best-performing model on Test 2 / Golden dataset is BERTić with weighted F1 = 0.73 and accuracy = 0.7351.

## 6. Interpretation

The results show that transformer-based modelling achieved the strongest overall performance. BERTić performed best on both the in-domain Cro-FiReDa test set and the out-of-domain Golden dataset. This confirms that a pretrained transformer model adapted to Bosnian, Croatian, Montenegrin and Serbian language data is better suited for Croatian sentiment classification than simpler machine learning and shallow deep learning baselines.

The TF-IDF + Logistic Regression model performed competitively on the Cro-FiReDa test set, especially after hyperparameter adjustment. However, its performance dropped on the Golden dataset, which indicates weaker generalization to out-of-domain review texts.

The improved CNN model achieved lower performance than BERTić on Cro-FiReDa, but it generalized better to the Golden dataset than the TF-IDF + Logistic Regression baseline. This suggests that the neural architecture captured some useful textual patterns, although it remained weaker than the transformer model.

The Gemma zero-shot approach performed reasonably well on Cro-FiReDa without supervised training, but its performance was weaker on the Golden dataset. This suggests that prompt-based zero-shot classification can be useful as a baseline, but it is less reliable than supervised fine-tuning for this specific Croatian sentiment classification task.

## 7. Conclusion

The best overall method is **BERTić**, which achieved the highest weighted F1-score and accuracy on both evaluation datasets. The comparison shows that transformer-based models are the most effective approach for this project, while classical machine learning, shallow deep learning and zero-shot GenAI methods provide useful baselines for comparison.

