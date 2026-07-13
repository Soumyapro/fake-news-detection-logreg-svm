# Fake News Detection using NLP & Machine Learning

A machine learning pipeline that classifies news articles as **Real** or **Fake** using TF-IDF vectorization and classical ML models (Logistic Regression, Linear SVM).

## Overview

This project builds and compares two text classification models to detect fake news articles. It includes full preprocessing, model training, performance evaluation, manual testing on custom examples, and an honest look at a data leakage issue discovered during evaluation.

## Dataset

[Fake News Detection Using Machine Learning](https://www.kaggle.com/datasets/subho117/fake-news-detection-using-machine-learning) (Kaggle)

- `True.csv` — real news articles
- `Fake.csv` — fake news articles
- Combined dataset: ~44,900 articles, labeled `1` (Real) / `0` (Fake)

## Pipeline

1. **Data loading & labeling** — load both CSVs, assign labels, concatenate
2. **Shuffling** — randomize row order to prevent class-order bias
3. **Cleaning** — drop unused columns (`title`, `subject`, `date`), keep `text` and `label`
4. **Text preprocessing** — lowercase, strip URLs, strip HTML tags, remove punctuation
5. **Feature extraction** — TF-IDF vectorization
6. **Train/test split** — 70/30 split
7. **Model training** — Logistic Regression and Linear SVM
8. **Evaluation** — accuracy, precision, recall, F1-score, confusion matrices
9. **Manual testing** — predictions on custom, unseen headlines/articles

## Results

| Model | Train Accuracy | Test Accuracy | Errors (of 13,470) |
|---|---|---|---|
| Logistic Regression | 99.08% | 98.67% | 179 |
| **Linear SVM** | 99.95% | **99.49%** | **69** |

Linear SVM outperformed Logistic Regression across the board, cutting total misclassifications by roughly 61%. The train-test accuracy gap stayed under 0.5% for both models, indicating neither model is meaningfully overfitting.

### Confusion Matrices

Both models were evaluated on the full test set with confusion matrices comparing false positive and false negative rates. Linear SVM produced fewer errors in both directions — fewer fake articles misclassified as real, and fewer real articles misclassified as fake.

## Manual Testing

The notebook includes a `predict_news()` function for testing the model on custom text outside the dataset:

```python
predict_news("The Federal Reserve raised interest rates by a quarter point on Wednesday...")
predict_news("Doctors are stunned after discovering that drinking hot lemon water...")
```

## Tech Stack

- Python
- pandas, numpy
- scikit-learn (TF-IDF, Logistic Regression, LinearSVC, metrics)
- matplotlib, seaborn (visualization)

## Future Improvements

- Test on an external, out-of-domain fake news dataset to measure true generalization
- Try additional models (Naive Bayes, Passive Aggressive Classifier) and ensemble approaches
- Explore transformer-based embeddings (e.g. BERT) as a stronger alternative to TF-IDF

## How to Run

1. Download dataset from the [dataset link](https://www.kaggle.com/datasets/subho117/fake-news-detection-using-machine-learning) and place in the project directory
2. Install dependencies:
   ```
   pip install numpy pandas scikit-learn matplotlib seaborn
   ```
3. Run the notebook cells in order
