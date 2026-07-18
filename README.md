# B2B Customer Churn Prediction: A Synthetic Control Framework for SMEs

## Overview
This repository contains a data-driven machine learning pipeline designed to predict business-to-business (B2B) customer churn for Small and Medium Enterprises (SMEs) in Malaysia. By utilizing a real-world dataset of corporate client records, the pipeline provides an explainable risk mitigation matrix optimized for resource-constrained enterprise systems.

## The Data Feasibility Challenge
Predictive modeling for this project faced a severe constraint due to data scarcity: an absolute absence of retained customer records. To resolve this single-class limitation, a synthetic control cohort was generated via profile bootstrapping to represent retained behaviors. The distribution fidelity of this generated data was subsequently verified using an adversarial Random Forest discriminator.

## Methodology & Pipeline Architecture
The analytics pipeline processes the data through the following engineering stages:
* **Zero-Contamination Preprocessing:** Missing value imputations were deferred until after the train-test split to prevent lookahead bias and statistical leakage.
* **Synthetic Cohort Generation:** Ordinal ratings were shifted using bounded stochastic perturbations to avoid deterministic constant offsets and scale compression.
* **Multicollinearity Control:** Derived features were engineered to replace raw ratings, ensuring the Variance Inflation Factor (VIF) for continuous baseline features remained safely below regulatory thresholds.
* **Model Evaluation:** Logistic Regression, Random Forest, and XGBoost classifiers were evaluated utilizing a Scikit-Learn pipeline embedded within a 5-fold, 3-repeat Stratified Cross-Validation framework.

## Key Results
* Comparative evaluations demonstrated that non-linear tree ensembles overfit the synthetic decision boundaries.
* The regularized Logistic Regression baseline generalized robustly to unseen data.
* The Logistic Regression model achieved a holdout Test AUC of 0.7603 and an optimal Brier score of 0.2036.

## Model Explainability
To ensure the pipeline is transparent and actionable, SHapley Additive exPlanations (SHAP) were integrated. SHAP values break down exactly which behaviors drive individual clients to churn, providing clear rationales for risk scores and eliminating the limitations of black-box models.

## Tech Stack
* **Language:** Python 3.11
* **Data Manipulation:** Pandas
* **Machine Learning:** Scikit-Learn, XGBoost
* **Interpretability:** SHAP
* **Visualization:** Matplotlib, Seaborn
