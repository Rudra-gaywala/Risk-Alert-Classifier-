
# Risk Alert Classification

A machine-learning practical project that builds a baseline risk classification model and evaluates techniques for handling class imbalance.

## Project overview

The notebook `pr_3.ipynb` uses a risk-alert dataset with a target column named `risk_status`. It separates the input features from the target, splits the data into training and testing sets, preprocesses numerical and categorical columns, trains a logistic regression classifier, and evaluates its predictions. It then applies four class-balancing methods and compares their results.

## Dataset

Expected dataset file:

`Risk_Alert_Classifier_Dataset_4600 - Risk_Alert_Classifier_Dataset_4600.csv.csv`

Update the CSV file path in your notebook before running it.

### Target and features

- **Target:** `risk_status`
- **Excluded columns:** `customer_id` and `last_transaction_date`
- **Input features:** all remaining columns

## Methods used in the notebook

### Data preparation and baseline model

- pandas and NumPy for data handling
- train-test split
- KNNImputer for missing numerical values
- SimpleImputer for missing categorical values
- StandardScaler for numerical feature scaling
- OneHotEncoder for categorical feature encoding
- ColumnTransformer and Pipeline for preprocessing

### Baseline evaluation

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score
- Type-I Error (False Positive)
- Type-II Error (False Negative)

### Class-imbalance handling

1. **Random under-sampling** – reduces majority-class samples.
2. **Random over-sampling** – duplicates minority-class samples.
3. **SMOTE** – creates synthetic minority-class samples.
4. **ADASYN** – generates synthetic samples for difficult minority-class examples.

The models are evaluated on the original test set using recall, F1-score and AUC-ROC.

## Requirements

Python 3.9 or later is recommended.

Install the required packages:

```bash
python -m pip install numpy pandas scikit-learn imbalanced-learn matplotlib jupyter
```

## How to run

1. Download or clone this project.
2. Place the CSV dataset in the project folder.
3. Install the required packages.
4. Open the terminal and run:

```bash
jupyter notebook
```

5. Open `pr_3.ipynb`.
6. Run all notebook cells from top to bottom.

## Important notes

- Preprocessing is fitted on training data and applied to testing data.
- Resampling is applied only to training data.
- The test dataset remains unchanged.
- Model performance depends on the dataset and selected parameters.
- False negatives are important in risk prediction because they represent risky customers classified as safe.
- Recall, F1-score and AUC-ROC should be considered when selecting the final model.

## Project files

```text
project-folder/
├── pr_3.ipynb
├── Risk_Alert_Classifier_Dataset_4600 - Risk_Alert_Classifier_Dataset_4600.csv.csv
└── README.md
```

## Evaluation guidance

Compare models using:

- **Recall:** correctly identified actual risk cases.
- **F1-score:** balance between precision and recall.
- **AUC-ROC:** ability to distinguish between both classes.
- **False negatives:** risky cases incorrectly predicted as non-risk.

Choose the final model based on business requirements and the cost of false positives and false negatives.
