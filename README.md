# Vendor Invoice Intelligence System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![ML](https://img.shields.io/badge/ML-Linear%20%7C%20Logistic%20%7C%20Random%20Forest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

An end-to-end machine learning system that predicts freight costs and flags high-risk invoices — so procurement teams spend time where it actually matters.

---

## The Problem

Two silent inefficiencies drain logistics operations every day:

1. **No cost benchmark** — teams accept freight quotes without knowing if the price is reasonable.
2. **Undifferentiated invoice review** — AP teams treat a ₹5,000 invoice the same as a ₹50,00,000 one.

This system solves both.

---

## What It Does

### 🔵 Freight Cost Prediction
Uses **Linear Regression** to predict expected freight cost based on order size. Gives procurement a data-backed benchmark before negotiations begin.

### 🔴 High-Risk Invoice Flagging
Uses **Logistic Regression** and **Random Forests** to score every invoice for risk. High-risk invoices are flagged automatically — concentrating human effort on vendors that actually matter.

---

## Repository Structure

```
Vendor-Invoice-Intelligence-System/
│
├── App.py                        # Streamlit app — main interface for predictions & flagging
│
├── Models/                       # Trained ML models (serialized)
│   ├── freight_cost_model.pkl
│   └── invoice_risk_model.pkl
│
├── NoteBooks/                    # Jupyter Notebooks for EDA, training & experimentation
│   ├── freight_cost_prediction.ipynb
│   └── invoice_risk_flagging.ipynb
│
├── Scripts/                      # General data processing scripts
│
├── Invoice_Flagging_Scripts/     # Logic dedicated to high-risk invoice detection
│
├── inference/                    # Modules to run models on new incoming data
│
└── anaconda_projects/db/         # Database integration & Anaconda environment config
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.8+ |
| ML Models | Scikit-learn (Linear Regression, Logistic Regression, Random Forest) |
| App | Streamlit |
| Experimentation | Jupyter Notebooks |
| Environment | Anaconda |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/pantpiyush11/Vendor-Invoice-Intelligence-System.git
cd Vendor-Invoice-Intelligence-System
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run App.py
```

---

## Models

### Freight Cost Prediction — Linear Regression
- **Input:** Order size / quantity
- **Output:** Predicted freight cost (₹)
- **Use case:** Benchmark quotes before procurement negotiations

### Invoice Risk Flagging — Logistic Regression + Random Forest
- **Input:** Invoice features (vendor, amount, frequency, anomalies, etc.)
- **Output:** Risk score + binary flag (High Risk / Normal)
- **Use case:** Prioritize human review on invoices that carry real financial risk

---

## Pipeline Design

The system is built as a **modular pipeline** — each stage (data ingestion, preprocessing, inference, output) is decoupled and independently reusable. This makes it easy to swap models, update training data, or plug in new vendors without touching the rest of the system.

---

## Author

**Piyush Pant**
[GitHub](https://github.com/pantpiyush11) · [LinkedIn](https://www.linkedin.com/in/pantpiyush11)

---

## License

This project is licensed under the MIT License.
