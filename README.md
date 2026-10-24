# Credit Risk Prediction Using Machine Learning

Predicting loan default from applicant and loan information, and comparing nine classification models.

## Project Overview

The notebook builds a full machine learning pipeline on the Kaggle **Credit Risk Dataset** (32,581 loan records, 12 columns). The target is `loan_status` (1 = default).

It covers:
- Exploratory data analysis and visualisation
- Duplicate removal and outlier handling
- Feature engineering and categorical encoding
- Train-only imputation and scaling (no data leakage)
- Class balancing with Random Over-Sampling (training set only)
- Training and comparing 9 classifiers

## Dataset

Download `credit_risk_dataset.csv` from Kaggle:
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

A copy of the file is already included in the `data/` folder. The notebook loads it with:

```python
df = pd.read_csv("data/credit_risk_dataset.csv")
```

## Preprocessing

1. Remove 165 duplicate records.
2. Remove implausible ages (age ≥ 90).
3. Keep only records where employment length is below `age − 15`.
4. Create `income_group`, `loan_amnt_group` and `loan_to_income`.
5. One-hot encode categorical variables (35 model features in total).
6. Stratified 80/20 train-test split (24,628 train / 6,158 test; 21.6% of test records are defaults).
7. Fit the imputer and scaler on the training data only, then apply them to the test data.
8. Balance the training set with `RandomOverSampler`.

## Results

All models were evaluated on the same held-out test set (1,332 defaults out of 6,158 records), sorted by ROC-AUC.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 0.930 | 0.873 | 0.794 | 0.831 | **0.950** |
| Random Forest | **0.936** | **0.935** | 0.758 | **0.837** | 0.938 |
| Gradient Boosting | 0.889 | 0.720 | 0.798 | 0.757 | 0.933 |
| SVM | 0.887 | 0.721 | 0.782 | 0.750 | 0.912 |
| Neural Network | 0.869 | 0.671 | 0.770 | 0.717 | 0.908 |
| K-Nearest Neighbors | 0.826 | 0.572 | 0.785 | 0.662 | 0.891 |
| Logistic Regression | 0.827 | 0.569 | **0.812** | 0.670 | 0.888 |
| Naive Bayes | 0.846 | 0.704 | 0.499 | 0.584 | 0.867 |
| Decision Tree | 0.898 | 0.754 | 0.785 | 0.769 | 0.857 |

### Key Findings

* **XGBoost ranked best on ROC-AUC (0.950)**, and **Random Forest had the best accuracy (0.936), precision (0.935) and F1 (0.837).**
* **Logistic Regression had the highest recall (0.812)** but the lowest precision of all models (0.569), so it catches more defaulters but also flags many good borrowers.
* **Naive Bayes missed about half of the defaulters** (recall 0.499).
* For credit risk, recall and precision matter alongside accuracy, because missing a risky borrower and rejecting a good one have different costs.

## Limitations

* Results come from a single train-test split; cross-validation has not been done yet.
* `loan_to_income` is very close to the existing `loan_percent_income` column, so it adds little new information.
* The employment-length filter also removes rows where employment length is missing.
* Models use fixed settings and no systematic hyperparameter tuning is shown in the notebook.

## How to Run

```bash
git clone https://github.com/priya-39/credit-risk-prediction.git
cd credit-risk-prediction
pip install -r requirements.txt
jupyter notebook credit-risk.ipynb
```

On Google Colab, upload the dataset to a `data/` folder first.

## Project Structure

```text
credit-risk-prediction/
├── credit-risk.ipynb
├── README.md
├── requirements.txt
└── data/
    ├── README.md
    └── credit_risk_dataset.csv
```

## Future Improvements

- Use `Pipeline` and `ColumnTransformer` for safer, reproducible preprocessing.
- Add cross-validation and systematic hyperparameter tuning.
- Plot ROC and precision-recall curves and analyse feature importance.
- Tune the decision threshold when recall of high-risk borrowers is the main goal.

## Disclaimer

This project is for educational and portfolio purposes only and should not be used as the basis for real lending decisions.
