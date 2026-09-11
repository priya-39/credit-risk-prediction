# Credit Risk Prediction Using Machine Learning

## Project Overview

This project builds a machine learning pipeline to predict **loan default / credit risk** using applicant and loan-related information.

The notebook performs:
- Exploratory Data Analysis (EDA)
- Missing-value inspection and imputation
- Duplicate removal
- Basic data cleaning and outlier handling
- Feature engineering
- Categorical-variable encoding
- Feature scaling
- Class balancing using Random Over-Sampling
- Training and comparison of multiple classification models
- Evaluation using Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and Classification Report

## Dataset

The project uses the **Credit Risk Dataset** containing information such as:
- Person age
- Person income
- Home ownership
- Employment length
- Loan intent
- Loan grade
- Loan amount
- Interest rate
- Loan status
- Historical credit information

> The dataset is not included in this repository. Download it separately and update the dataset path in the notebook.

## Models Used

The following classification algorithms are implemented:

1. Gaussian Naive Bayes
2. Decision Tree
3. Random Forest
4. K-Nearest Neighbors (KNN)
5. Logistic Regression
6. Support Vector Machine (SVM)
7. Multi-Layer Perceptron (Neural Network)
8. Gradient Boosting
9. XGBoost

## Feature Engineering

Two categorical grouping features are created:

- `income_group` – groups applicants according to annual income
- `loan_amnt_group` – groups loans according to loan amount

A numerical feature is also created:

- `loan_to_income` = loan amount / person income

## Data Preprocessing

The notebook applies the following preprocessing steps:

1. Remove duplicate records
2. Remove implausible age values
3. Remove records where employment length is inconsistent with age
4. Create engineered features
5. One-hot encode categorical variables
6. Split data into training and testing sets using stratification
7. Impute missing values using the training data
8. Standardize features using the training data
9. Balance the training set using `RandomOverSampler`

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Classification Report

For credit-risk prediction, **Recall and Precision should be considered alongside Accuracy**, because the cost of incorrectly classifying a risky borrower can be important.

## How to Run

### Option 1: Google Colab

1. Open the notebook in Google Colab.
2. Upload the dataset.
3. Change the dataset path in the first data-loading cell.
4. Run the notebook from top to bottom.

### Option 2: Jupyter Notebook

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn imbalanced-learn xgboost
```

Then open:

```text
credit-risk.ipynb
```

## Important Note About the Dataset Path

The original notebook contains a Kaggle-specific path:

```python
/kaggle/input/datasets/priyatak0723/credit-risk-dataset/credit_risk_dataset.csv
```

This path will not work automatically for other users.

For a GitHub project, use a local/project-relative path such as:

```python
df = pd.read_csv("data/credit_risk_dataset.csv")
```

or upload the dataset separately in Google Colab and update the path.

## Project Structure

```text
credit-risk-prediction/
│
├── credit-risk.ipynb
├── README.md
├── requirements.txt
└── data/
    └── README.md
```

## Dataset

The dataset is not included in this repository.

Please download the Credit Risk Dataset from Kaggle:

https://www.kaggle.com/datasets/laotse/credit-risk-dataset

After downloading and extracting the dataset, place:

`credit_risk_dataset.csv`

inside:

`data/`

The notebook loads the dataset using:

```python
df = pd.read_csv("data/credit_risk_dataset.csv")
```
## Reproducibility

The train-test split and most models use fixed random states where applicable. The experiment uses an 80/20 stratified train-test split.

## Future Improvements

- Use `Pipeline` and `ColumnTransformer` to make preprocessing safer and reproducible.
- Perform cross-validation.
- Tune hyperparameters systematically.
- Compare models in a single results table.
- Plot ROC and Precision-Recall curves.
- Analyze feature importance and model interpretability.
- Evaluate threshold tuning when recall of high-risk borrowers is the primary objective.

## Disclaimer

This project is for educational and portfolio purposes. It should not be used as the sole basis for real-world lending or financial decisions.
