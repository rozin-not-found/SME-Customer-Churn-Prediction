# B2B Customer Churn Prediction: A Synthetic Control Framework

## Project Overview
This repository contains a data-driven machine learning pipeline designed to predict business-to-business (B2B) customer churn for Small and Medium Enterprises (SMEs) in Malaysia[cite: 2]. By utilizing a real-world dataset of corporate client records, the pipeline provides an explainable risk mitigation matrix optimized for resource-constrained enterprise systems[cite: 2].

## The Data Feasibility Challenge
Predictive modeling for this project faced a severe constraint due to data scarcity: an absolute absence of retained customer records[cite: 2]. To resolve this single-class limitation, a synthetic control cohort was generated via profile bootstrapping to represent retained behaviors[cite: 2]. The distribution fidelity of this generated data was subsequently verified using an adversarial Random Forest discriminator[cite: 2].

## Methodology & Pipeline Architecture
The analytics pipeline processes the data through the following engineering stages:
* **Zero-Contamination Preprocessing:** Missing value imputations were deferred until after the train-test split to prevent lookahead bias and statistical leakage[cite: 2].
* **Synthetic Cohort Generation:** Ordinal ratings were shifted using bounded stochastic perturbations to avoid deterministic constant offsets and scale compression[cite: 2].
* **Multicollinearity Control:** Derived features were engineered to replace raw ratings, ensuring the Variance Inflation Factor (VIF) for continuous baseline features remained safely below regulatory thresholds[cite: 2].
* **Model Evaluation:** Logistic Regression, Random Forest, and XGBoost classifiers were evaluated utilizing a Scikit-Learn pipeline embedded within a 5-fold, 3-repeat Stratified Cross-Validation framework[cite: 2].

## Key Results
* Comparative evaluations demonstrated that non-linear tree ensembles overfit the synthetic decision boundaries[cite: 2].
* The regularized Logistic Regression baseline generalized robustly to unseen data[cite: 2].
* The Logistic Regression model achieved a holdout Test AUC of 0.7603 and an optimal Brier score of 0.2036[cite: 2].

## Model Explainability
To ensure the pipeline is transparent and actionable, SHapley Additive exPlanations (SHAP) were integrated[cite: 2]. SHAP values break down exactly which behaviors drive individual clients to churn, providing clear rationales for risk scores and eliminating the limitations of black-box models[cite: 2].

## Tech Stack
* **Language:** Python 3.11[cite: 2]
* **Data Manipulation:** Pandas[cite: 2]
* **Machine Learning:** Scikit-Learn, XGBoost[cite: 2]
* **Interpretability:** SHAP[cite: 2]
* **Visualization:** Matplotlib, Seaborn[cite: 2]
