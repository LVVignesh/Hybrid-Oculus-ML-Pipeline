# 🛒 Hybrid-Oculus: Online Shoppers Purchase Prediction
**An End-to-End Hybrid Machine Learning Pipeline (Supervised + Unsupervised + XAI)**

---

## 📌 Project Overview
**Hybrid-Oculus-ML** is a sophisticated machine learning framework designed to decode and predict online consumer intent. The "Oculus" (Eye) represents the project's focus on **Explainable AI (XAI)**—moving beyond "black-box" predictions to understand exactly why a customer is likely to convert.

This project combines supervised learning, unsupervised behavioral discovery, and advanced model validation into a single production-style workflow.

## 🎯 Problem Statement
Given a user’s browsing session data on an e-commerce website, can we predict if the user will make a purchase? This is a binary classification problem with a significant class imbalance (~15% positive class), making sophisticated evaluation metrics critical.

## 🧠 Technical Architecture (The "Hybrid" Approach)
The system follows a three-stage hybrid architecture:

1. **Unsupervised Discovery:** Employs **PCA** for noise reduction and **K-Means Clustering** to segment users into behavioral archetypes.
2. **Supervised Intelligence:** Injects discovered cluster labels back into a tuned **XGBoost Classifier** to enhance predictive accuracy.
3. **Stability & Transparency:** Uses **Stratified K-Fold Cross-Validation** and **SHAP** to map the global and local impact of features.

## 📊 Performance Metrics

| Metric | Value |
| :--- | :--- |
| **Mean ROC-AUC (5-Fold CV)** | **0.933** |
| **Test ROC-AUC** | **0.924** |
| **CV Standard Deviation** | **0.005** |
| **Precision (Purchase Class)** | **0.75** |

## 🛠️ Key Features

* **Robust Preprocessing:** Utilizes `RobustScaler` to handle outliers in behavioral session data.

* **Automated Tuning:** Employs `RandomizedSearchCV` to mathematically optimize XGBoost hyperparameters.

* **Explainable AI (XAI):** Integrated **SHAP** to satisfy the "Right to Explanation" requirements in modern AI ethics.

* **Stratified Validation:** 5-Fold Stratified CV confirms the model does not "memorize" the majority class.

## 📂 Project Structure

Hybrid-Oculus-ML-Pipeline/
│
├── Online_shoppers Project.py     
├── online_shoppers_intention.csv 
├── .gitignore                     
├── LICENSE                        
└── README.md                     

Author: LV Vignesh | Status: Production-Ready | License: MIT