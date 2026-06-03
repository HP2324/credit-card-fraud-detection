# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using both
unsupervised anomaly detection and supervised classification techniques. The project
explores and compares multiple algorithms on a highly imbalanced real-world dataset,
covering the full pipeline from exploratory data analysis to model evaluation.

---

## Features

- Exploratory data analysis with visualizations (transaction distributions, time vs amount, correlation heatmap)
- Unsupervised anomaly detection using Isolation Forest, Local Outlier Factor, and One-Class SVM
- Supervised classification using Logistic Regression, Decision Tree, and Random Forest
- Handles severe class imbalance (0.172% fraud) via undersampling and SMOTE oversampling
- Side-by-side model comparison with accuracy, precision, recall, and F1 metrics
- Saves the best performing model using joblib for reuse

---

## Technologies Used

**Machine Learning & Data**
- Python 3.7+
- NumPy, Pandas
- Scikit-learn
- imbalanced-learn (SMOTE)
- SciPy

**Visualization**
- Matplotlib
- Seaborn

**Environment**
- Jupyter Notebook

---

## Project Structure

```
credit-card-fraud-detection/
├── Anomaly-Detection.ipynb                    # Unsupervised anomaly detection
├── Supervised_Learning.ipynb                  # Supervised classification
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Ignores dataset, checkpoints, etc.
└── README.md
```

---

## Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it in the root directory before running the notebooks.

| Property | Details |
|---|---|
| Source | Worldline & ULB Machine Learning Group |
| Transactions | 284,807 |
| Fraud cases | 492 (0.172% — highly imbalanced) |
| Features | V1–V28 (PCA-transformed), Time, Amount |
| Target | `Class` (0 = Normal, 1 = Fraud) |

> Due to confidentiality, the original features are replaced with PCA components V1–V28.
> Only `Time` and `Amount` retain their original form.

---

## Approach

This project tackles fraud detection from two distinct angles:

### 1. Unsupervised Anomaly Detection (`Anomaly-Detection.ipynb`)
Treats fraud as an anomaly without relying on labeled training data. This approach is
more practical in real-world pipelines where labeled fraud data is scarce or arrives
with a delay. Three algorithms are compared:

- **Isolation Forest** — isolates anomalies by randomly partitioning data using
  decision trees. Anomalies require fewer splits to isolate, resulting in shorter
  path lengths and lower anomaly scores. Uses a `contamination` parameter set to
  the true fraud ratio in the dataset.
- **Local Outlier Factor (LOF)** — detects outliers based on local density deviation.
  Points with significantly lower density than their neighbors are flagged as anomalies.
  Uses `n_neighbors=20`.
- **One-Class SVM** — learns a boundary around normal transactions using an RBF kernel
  and flags anything outside it as anomalous.

### 2. Supervised Classification (`Supervised_Learning.ipynb`)
Treats fraud detection as a binary classification problem using labeled data. Addresses
class imbalance through two strategies — undersampling the majority class and SMOTE
oversampling — and compares three classifiers:

- **Logistic Regression** — linear baseline model
- **Decision Tree** — non-linear, interpretable model
- **Random Forest** — ensemble of decision trees, strongest overall performer

---

## Results

### Unsupervised Anomaly Detection

| Model | Accuracy | Errors | Fraud Recall |
|---|---|---|---|
| Isolation Forest | 99.74% | 73 | ~27% |
| Local Outlier Factor | 99.65% | 97 | ~2% |
| One-Class SVM | 70.09% | 8,516 | 0% |

**Isolation Forest** is the clear winner — fewest errors and the highest fraud recall
by a large margin.

### Supervised Classification — Undersampling (473 normal + 473 fraud samples)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 93.68% | 0.9688 | 0.9118 | 0.9394 |
| Decision Tree | 87.89% | 0.8762 | 0.9020 | 0.8889 |
| Random Forest | 93.16% | 0.9684 | 0.9020 | 0.9340 |

### Supervised Classification — SMOTE Oversampling (275,190 samples per class)

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 94.38% | 0.9732 | 0.9127 | 0.9419 |
| Decision Tree | 99.83% | 0.9977 | 0.9990 | 0.9983 |
| Random Forest | 99.99% | 0.9998 | 1.0000 | 0.9999 |

---

## Key Findings

- With SMOTE oversampling, Random Forest achieves near-perfect performance —
  99.99% accuracy, 0.9998 precision, and 1.0 recall — making it the strongest
  supervised model by a significant margin.
- Undersampling (473 samples per class) trains fast but loses information from
  the majority class, resulting in lower overall accuracy (~87–94%) compared to
  SMOTE (~94–99.99%).
- SMOTE generates synthetic fraud samples to fully balance the dataset (275,190
  per class), giving models far more signal to learn from, which explains the
  dramatic jump in Decision Tree and Random Forest performance.
- Isolation Forest achieves ~27% fraud recall with zero labeled training data,
  making it the most practical approach in real-world scenarios where fraud labels
  are unavailable or delayed.
- Raw accuracy is a misleading metric on this dataset — a model that always
  predicts "no fraud" would still achieve 99.8% accuracy due to the 0.172% fraud
  rate. Precision, Recall, F1, and AUPRC are more meaningful.
- The best performing model (Random Forest trained on SMOTE data) is saved using
  joblib for reuse without retraining.

---

## Installation

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd credit-card-fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**
Get `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it in the root directory.

**4. Launch Jupyter**
```bash
jupyter notebook
```

Run `Anomaly-Detection.ipynb` for unsupervised results, then
`Supervised_Learning.ipynb` for supervised classification results.

---

## Notebooks

### `Anomaly-Detection.ipynb`
| Section | Description |
|---|---|
| EDA | Class distribution, fraud vs normal amount histograms, time vs amount scatter plots, correlation heatmap |
| Preprocessing | 10% random sample for model efficiency, outlier fraction calculation |
| Modeling | Isolation Forest, LOF, One-Class SVM trained and evaluated |
| Evaluation | Accuracy, classification report (precision, recall, F1) per model |

### `Supervised_Learning.ipynb`
| Section | Description |
|---|---|
| Preprocessing | Feature scaling (StandardScaler on Amount), duplicate removal, Time column dropped |
| Imbalance Handling | Undersampling to 473 normal samples; SMOTE oversampling |
| Modeling | Logistic Regression, Decision Tree, Random Forest trained and compared under both strategies |
| Evaluation | Accuracy, precision, recall, F1 per model; bar chart comparison |
| Model Saving | Best model saved to `model_credit.pkl` using joblib with a reusable prediction function |

---

## References

- [Kaggle Dataset — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Research collaboration between Worldline and the Machine Learning Group at ULB
  (Université Libre de Bruxelles)
- [Isolation Forest — Liu et al., 2008](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
- [Local Outlier Factor — Breunig et al., 2000](https://dl.acm.org/doi/10.1145/342009.335388)