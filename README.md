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

---

## 📊 Performance Metrics & Visualizations

### 1. Model Reliability (ROC Curve)
The model achieved a high level of discriminative power, successfully distinguishing between buyers and non-buyers.
* **Test ROC-AUC: 0.924**

![ROC Curve](https://raw.githubusercontent.com/LVVignesh/Hybrid-Oculus-ML-Pipeline/main/plots/roc_curve.png)

### 2. Prediction Quality (Confusion Matrix)
The confusion matrix shows strong performance in predicting the majority class, with targeted accuracy for the minority (purchase) class.

![Confusion Matrix](https://raw.githubusercontent.com/LVVignesh/Hybrid-Oculus-ML-Pipeline/main/plots/confusion_matrix.png)

### 3. Precision-Recall Trade-off
Given the class imbalance, the PR Curve demonstrates how the model maintains precision even as we increase recall.

![Precision-Recall Curve](https://raw.githubusercontent.com/LVVignesh/Hybrid-Oculus-ML-Pipeline/main/plots/precision_recall_curve.png)

### 4. Explainable AI (SHAP Summary)
Using SHAP, we "opened the black box" to see which features (Page Views, Duration, Exit Rates) most heavily influence the final purchase prediction.

![SHAP Summary](https://raw.githubusercontent.com/LVVignesh/Hybrid-Oculus-ML-Pipeline/main/plots/shap_summary.png)

---

## 🛠️ Key Features
* **Robust Preprocessing:** Utilizes `RobustScaler` to handle outliers in behavioral session data.
* **Automated Tuning:** Employs `RandomizedSearchCV` to mathematically optimize XGBoost hyperparameters.
* **Explainable AI (XAI):** Integrated **SHAP** to satisfy transparency requirements in modern AI ethics.
* **Stratified Validation:** 5-Fold Stratified CV confirms the model does not "memorize" the majority class.

## 📂 Project Structure
```text
Hybrid-Oculus-ML-Pipeline/
│
├── Online_shoppers Project.py     # Master end-to-end pipeline script
├── online_shoppers_intention.csv  # Raw dataset
├── .gitignore                     # Python-specific ignore rules
├── LICENSE                        # MIT Open Source License
└── README.md                      # Project documentation