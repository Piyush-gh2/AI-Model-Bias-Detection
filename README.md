# AI Model Bias Detection

## 🔬 An Experimental Study of Fairness in Machine Learning

## 📌 Project Overview

**AI Model Bias Detection** is an experimental AI research project designed to investigate whether a machine learning model produces different prediction outcomes across different groups.

The project uses a **synthetic loan-approval dataset** and a Logistic Regression classification model. Instead of focusing only on overall model accuracy, the project evaluates model performance at both the overall and group levels.

The objective is to demonstrate a simple and reproducible approach to studying **AI fairness, model bias, and Responsible AI**.

> **Note:** The dataset used in this project is synthetic and created for educational and research experimentation. `Group_A` and `Group_B` are artificial groups and do not represent real demographic populations.

---

## 🎯 Research Objective

The main objective is:

> **To investigate whether a machine learning model performs consistently across different groups.**

### Research Question

> **Does the ML model produce similar prediction performance across synthetic groups?**

### Hypothesis

A model can achieve good overall performance while still showing differences in prediction performance across groups.

---

## 🧠 Research Concept

Traditional ML evaluation generally focuses on:

```text
Model
  ↓
Predictions
  ↓
Accuracy
```

This project adds group-level analysis:

```text
                    ML Model
                       ↓
                  Predictions
                       ↓
            ┌──────────┴──────────┐
            ↓                     ↓
     Overall Evaluation     Group Evaluation
            ↓                     ↓
      Accuracy/F1          Group A / Group B
                                  ↓
                           Fairness Analysis
                                  ↓
                            Performance Gap
```

---

## 🏗️ System Architecture

```text
                  ┌─────────────────────┐
                  │      Dataset       │
                  │   Synthetic Data   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Data Preprocessing  │
                  │                     │
                  │ Scaling & Splitting │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Logistic Regression │
                  │       Model         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │     Predictions     │
                  └──────────┬──────────┘
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
       ┌─────────────────┐       ┌──────────────────┐
       │ Overall Metrics │       │ Group-Level      │
       │                 │       │ Evaluation       │
       │ Accuracy        │       │                  │
       │ Precision       │       │ Group A          │
       │ Recall          │       │ Group B          │
       │ F1 Score        │       │                  │
       └─────────────────┘       └─────────┬────────┘
                                           │
                                           ▼
                                  ┌──────────────────┐
                                  │ Fairness Analysis│
                                  │                  │
                                  │ Performance Gap  │
                                  └─────────┬────────┘
                                            │
                                            ▼
                                  Research Findings
```

---

## 📊 Dataset

The project uses a synthetic loan-approval dataset containing **1,000 records**.

### Features

| Feature            | Description                |
| ------------------ | -------------------------- |
| `group`            | Synthetic group identifier |
| `age`              | Applicant age              |
| `income`           | Annual income              |
| `credit_score`     | Synthetic credit score     |
| `debt_ratio`       | Debt-to-income style ratio |
| `employment_years` | Years of employment        |
| `approved`         | Target variable: 0 or 1    |

### Target

```text
0 → Not Approved
1 → Approved
```

The dataset is intentionally designed for experimentation and should **not** be used for real-world lending decisions.

---

## 🔬 Research Methodology

The project follows these steps:

### Step 1 — Data Collection

Use the synthetic dataset containing applicant and group information.

### Step 2 — Data Preprocessing

* Separate features and target.
* Keep the group attribute for fairness analysis.
* Split the data into training and testing sets.
* Standardize numerical features.

### Step 3 — Model Training

Train a Logistic Regression classification model.

### Step 4 — Prediction

Generate predictions using the test dataset.

### Step 5 — Overall Evaluation

Calculate:

* Accuracy
* Precision
* Recall
* F1-score

### Step 6 — Group-Level Evaluation

Calculate performance separately for each synthetic group.

### Step 7 — Fairness Analysis

Calculate differences between groups using metrics such as:

* Accuracy gap
* Positive prediction rate gap
* True positive rate gap

---

## 📐 Fairness Metrics

### Accuracy Gap

```text
Accuracy Gap =
Maximum Group Accuracy - Minimum Group Accuracy
```

### Positive Prediction Rate Gap

```text
Positive Prediction Rate Gap =
Maximum Group Positive Rate
-
Minimum Group Positive Rate
```

### True Positive Rate Gap

```text
TPR Gap =
Maximum Group TPR
-
Minimum Group TPR
```

Smaller gaps indicate more similar performance between the groups for that metric, but **no single metric proves that a model is fair or unfair**.

---

## 🧪 Experimental Design

The primary experiment evaluates:

```text
Synthetic Dataset
       ↓
Logistic Regression
       ↓
Predictions
       ↓
Overall Performance
       +
Group-Level Performance
       ↓
Fairness Analysis
```

An optional extension compares:

### Experiment A

Model trained with the synthetic `group` feature.

### Experiment B

Model trained without the `group` feature.

The objective is to investigate whether removing a group attribute necessarily eliminates performance differences.

---

## 📈 Results

After running the experiment, the project generates:

```text
results/
│
├── metrics.csv
└── fairness_report.csv
```

The actual experimental results should be reported here.

Example structure:

| Metric    |        Result |
| --------- | ------------: |
| Accuracy  | Actual Result |
| Precision | Actual Result |
| Recall    | Actual Result |
| F1 Score  | Actual Result |

### Group-Level Results

| Group   | Accuracy | Positive Rate |    TPR |
| ------- | -------: | ------------: | -----: |
| Group_A |   Actual |        Actual | Actual |
| Group_B |   Actual |        Actual | Actual |

> Results should be generated from the experiment and should not be replaced with fabricated values.

---

## 🔎 Research Questions

This project can be extended to investigate:

1. Can overall accuracy hide group-level performance differences?
2. Does removing the group attribute reduce performance disparities?
3. How do classification thresholds affect fairness metrics?
4. Can model performance and fairness be improved simultaneously?
5. Which fairness metric is most informative for this experimental dataset?

---

## 🛠️ Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Flask

### Machine Learning Model

**Logistic Regression**

---

## 📁 Project Structure

```text
ai-model-bias-detection/
│
├── data/
│   └── dataset.csv
│
├── preprocessing/
│   └── preprocess.py
│
├── models/
│   └── model.py
│
├── evaluation/
│   ├── metrics.py
│   └── fairness.py
│
├── experiments/
│   └── experiment.py
│
├── results/
│   ├── metrics.csv
│   └── fairness_report.csv
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Open the project

```bash
cd ai-model-bias-detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/macOS

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Research Experiment

From the project root:

```bash
python experiments/experiment.py
```

The experiment will:

1. Load the dataset.
2. Preprocess the data.
3. Train the Logistic Regression model.
4. Generate predictions.
5. Calculate overall metrics.
6. Calculate group-level metrics.
7. Calculate fairness gaps.
8. Save the results.

---

## 📊 Generated Results

After execution:

```text
results/metrics.csv
```

contains the overall model metrics.

```text
results/fairness_report.csv
```

contains group-level performance and fairness analysis.

---

## 🌐 Run the Web Application

Start the Flask application:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The application provides a simple interface for entering synthetic applicant information and obtaining a model prediction.

---

## ⚠️ Responsible AI Disclaimer

This project is intended **only for educational and research purposes**.

The dataset is synthetic and the model is not suitable for making real-world decisions about loans, employment, insurance, credit, or other high-impact domains.

Fairness analysis is context-dependent, and a small number of statistical metrics cannot fully determine whether an AI system is fair.

---

## 🚀 Future Improvements

Future versions could include:

* Fairness-aware machine learning
* Threshold optimization
* Fairlearn
* SHAP-based explanations
* Multiple classification models
* Larger datasets
* Intersectional group analysis
* Calibration analysis
* Statistical significance testing
* Bias mitigation techniques
* Interactive fairness dashboard

---

## 🎓 Research Contribution

This project demonstrates a simple experimental framework for studying **AI fairness and Responsible AI**.

The key idea is to move beyond overall model accuracy and investigate:

```text
Model Performance
       +
Group-Level Performance
       +
Fairness Metrics
       ↓
Responsible AI Analysis
```

---

## 👨‍💻 Author

**Piyush Mishra**

AI / GenAI | Machine Learning | Responsible AI | Python | Java

---

## ⭐ Conclusion

The project demonstrates how a machine learning model can be evaluated from both a **performance perspective** and a **fairness perspective**.

The experiment provides a foundation for further research into **AI bias detection, fairness measurement, and Responsible AI**.
