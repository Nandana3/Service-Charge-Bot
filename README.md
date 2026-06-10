# Kuwait Expat Service Entitlement & Dispute Risk Tool

A dual-interface Streamlit application serving two audiences: **expatriate employees** who want to know the indemnity their employer legally owes them under Kuwait Labour Law, and **employers** who want to assess the probability that an employee is likely to file a labour dispute.

---

## Overview

Expatriate workers in Kuwait are often unaware of what they are legally owed when their employment ends. Calculating indemnity is non-trivial: it depends on tenure, reason for leaving, job category, and salary — and errors or omissions by employers are common triggers for labour disputes.

**For employees:**
- **Know your rights** — compute the exact indemnity owed based on your employment details
- **No legal background needed** — plain-language output, self-serve interface

**For employers:**
- **Assess dispute risk** — predict the probability that an employee is likely to escalate to a formal labour dispute, enabling early intervention

---

## Features

### 1. Indemnity Calculator *(Core Feature)*
Computes end-of-service entitlements per Kuwait Labour Law rules based on:
- Length of service / tenure
- Reason for leaving (resignation, termination, contract expiry)
- Job category and industry
- Monthly salary (in KWD)

Outputs the indemnity amount the employer is legally obligated to pay.

### 2. Dispute Risk Predictor *(Supporting Feature)*
- Random Forest classifier that predicts the probability an employee is likely to file a labour dispute
- Designed for **employers** to proactively identify high-risk cases and intervene before escalation
- 85%+ validation accuracy on held-out test data

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML Model | Scikit-learn (Random Forest Classifier) |
| Data | Pandas, synthetic CSV dataset |
| App | Streamlit |
| Serialization | Joblib, Pickle |
| Language | Python 3.x |

---

## Project Structure

```
Kuwait_Project/
│
├── app3.py                        # Main Streamlit application
├── train_model.py                 # Model training script
├── generate_data.py               # Synthetic dataset generation
├── kuwait_expat_labor_data.csv    # Generated training dataset
├── dispute_model.joblib           # Trained Random Forest model
├── model_columns.pkl              # Saved feature columns for inference
└── create_flowchart.py            # Architecture flowchart generator
```

---

## Model Details

- **Algorithm:** Random Forest Classifier (Scikit-learn)
- **Features:** `reason_for_leaving`, `industry`, `job_category`, `monthly_salary_kwd`
- **Target:** `filed_dispute` (binary)
- **Validation Accuracy:** 85%+
- **Training Data:** 5,000+ synthetic records encoded with Kuwait Labour Law attributes

Real employee data was intentionally excluded. The synthetic dataset was engineered to reflect true regulatory coverage without confidentiality risk.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Nandana3/Kuwait_Project.git
cd Kuwait_Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the model
```bash
python generate_data.py   # regenerate synthetic dataset
python train_model.py     # retrain and save model
```

### 4. Launch the app
```bash
streamlit run app3.py
```

---

## Requirements

```
pandas
scikit-learn
streamlit
joblib
```

---

## Key Results

- Gives expatriate employees instant visibility into their legal entitlements — no lawyer required
- **85%+ validation accuracy** on dispute risk prediction
- Estimated **70% reduction** in ad-hoc legal queries after deployment to Glorosh's legal team
- Full system ownership: data engineering → ML modelling → Streamlit deployment

---

## Context

Built during an internship at **Glorosh International** (Apr–Jul 2025), where the legal team handled Kuwait expatriate labour disputes. The tool serves two distinct audiences: expatriate employees wanting to understand what indemnity they are legally owed, and employers wanting to anticipate and manage dispute risk before it escalates.

---

## Author

**Nandana M K** — M.Sc. Data Science, Reva University  
[LinkedIn](https://linkedin.com/in/nandanamk) | [GitHub](https://github.com/Nandana3)