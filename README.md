# 🚀 Uplift Targeting Engine: Maximizing Ad ROI with Causal AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Dataset: Criteo](https://img.shields.io/badge/Dataset-Criteo_13M-green.svg)](https://ailab.criteo.com/criteo-uplift-dataset/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Executive Summary
Traditional marketing systems target users who are *likely to buy*. This is inefficient because it wastes money on users who would have bought anyway (**"Sure Things"**).

This project implements a **Causal Uplift Model** (T-Learner) using 13.9 million rows of real-world randomized control trial (RCT) data. It predicts the **incremental impact** of an ad, allowing businesses to target only the **"Persuadables"**—users who convert *only because* of the intervention.

### 💰 Business ROI Decision Matrix
| Segment | Conversion if Treated | Conversion if Control | Recommended Action |
| :--- | :--- | :--- | :--- |
| **Persuadables** | High | Low | **Target** 🎯 |
| **Sure Things** | High | High | **Avoid** (Save $) |
| **Lost Causes** | Low | Low | **Avoid** (Save $) |
| **Sleeping Dogs** | Low | High | **CRITICAL AVOID** ⚠️ |

---

## 🛠️ Technical Architecture

```mermaid
graph TD
    %% Node Definitions
    Dataset(["Raw 13.9M Criteo Dataset"]):::data
    Split{Treatment Group?}:::decision
    ModelT[Train Model_T: XGBoost]:::model
    ModelC[Train Model_C: XGBoost]:::model
    Inference([New User Request]):::request
    PredT[Model_T Prediction]:::process
    PredC[Model_C Prediction]:::process
    Calc[Uplift Calculation]:::process
    ActionI[Target with Ad]:::target
    ActionJ[Silence / Suppress Ad]:::avoid

    %% Connection Logic
    Dataset --> Split
    Split -- "Yes (85%)" --> ModelT
    Split -- "No (15%)" --> ModelC
    
    Inference --> PredT
    Inference --> PredC
    PredT -- "P(T)" --> Calc
    PredC -- "P(C)" --> Calc
    
    Calc -- "Uplift > 0" --> ActionI
    Calc -- "Uplift &le; 0" --> ActionJ

    %% Style Classes
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#01579b
    classDef decision fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#7f6000
    classDef model fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#2e7d32
    classDef request fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#7b1fa2
    classDef process fill:#eceff1,stroke:#455a64,stroke-width:2px,color:#455a64
    classDef target fill:#c8e6c9,stroke:#2e7d32,stroke-width:4px,color:#1b5e20,stroke-dasharray: 5 5
    classDef avoid fill:#ffcdd2,stroke:#c62828,stroke-width:2px,color:#b71c1c
```

### 🧠 Methodology: The T-Learner
The system trains two independent gradient-boosted trees (XGBoost):
1.  **Treatment Model ($M_T$):** Learns behavior under exposure to ads.
2.  **Control Model ($M_C$):** Learns baseline organic behavior.

**Uplift Score calculation:**
$$\text{Uplift} = P(\text{Conversion} | \text{Treated}) - P(\text{Conversion} | \text{Control})$$

### 📊 Performance Metrics (Phase 4 Evaluation)
*   **Dataset Size:** 13,912,825 rows.
*   **ROC-AUC (Treatment):** **0.9584** (Excellent discrimination).
*   **Ranking Logic:** Qini Curve analysis confirms significant uplift gain in the top 2 deciles.

---

## 📂 Project Structure
```text
├── notebooks/          # EDA and Model Evaluation (Calibration, Qini, ROC)
├── models/             # Pre-trained XGBoost Models
├── evaluate_full_model.py # Evaluation pipeline
├── train_full_model.py   # Training pipeline
├── requirements.txt    # Project Dependencies
└── README.md           # This documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Python 3.9+, pip, and an active virtual environment.

### 2. Quick Install
```bash
git clone https://github.com/srinath2934/uplift-targeting-engine.git
cd uplift-targeting-engine
pip install -r requirements.txt
```

---

## 🤝 Acknowledgments
*   **Criteo AI Lab:** For providing the massive-scale open-source dataset.
*   **XGBoost Team:** For the industry-standard gradient boosting framework.

---
**Developed by [Srinath](https://github.com/srinath2934)**
