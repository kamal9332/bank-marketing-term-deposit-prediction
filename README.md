# Predicting term-deposit subscriptions using ML on 41,176 bank marketing records

Machine-Learning pipeline was developed comparing Logistic Regression, Random Forest and XGBoost, with SHAP Explainable AI used to derive key factors associated with customers subscribing to term-deposit.

## 1. Business Problem

Indiscriminate telephone marketing incur additional costs, cause inconvenience to customers and hurt their desire to receive future marketing. It can also cause employees to make numerous unsuccessful phone calls before receiving a positive response. This is evident with the Portuguese Bank dataset, where only 4640 out of 41.188 observations were customers that subscribed to a term deposit. 

This project aims to address the challenges the Portuguese Bank is currently experiecing by developing a predictive marketing systems that enable banks to focus on those customers who are more likely to subscribe. 

## 2. Dataset
- Source: UCI Portuguese Bank Marketing dataset (Moro et al., 2014)
- 41,188 rows → **41,176 after cleaning** (12 duplicates removed)
- **19 predictors**: customer, campaign and macroeconomic features
- Target: subscribed to a term deposit (yes/no). **Imbalanced**: only about 11.27% subscribed.

## 3. Method
Framework: **CRISP-DM.**
- Dropped `duration` to prevent data leakage.
- Engineered `previously_contacted` from `pdays`.
- Stratified 60/20/20 train/validation/test split.
- Preprocessing fitted on training data only (one-hot encoding and scaling for LR, ordinal encoding for RF and XGBoost).
- SMOTE applied inside the CV folds only.
- 5-fold CV and GridSearchCV for hyperparameter tuning.

## 4. Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.8817** | **0.4782** | 0.5442 | **0.5091** | 0.7967 |
| Random Forest | 0.8798 | 0.4708 | 0.5377 | 0.5020 | **0.7977** |
| Logistic Regression | 0.8299 | 0.3590 | **0.6487** | 0.4622 | 0.7965 |

XGBoost had the highest F1 (the primary evaluation metric for imbalanced data). Logistic Regression identified the most subscribers (highest recall) but made more wasted calls. ROC-AUC was about 0.80 for all three models.

## 5. Key Insights and Business Recommendations
Top SHAP Features: `nr.employed` (0.447), `contact` (0.228), `emp.var.rate` (0.154), `poutcome` (0.147).

**Prioritise customers with a successful previous campaign.** They subscribed at 65.11%, against 8.83% for customers with no previous campaign.
- **Favour cellular contact.** It subscribed at 14.74%, against 5.23% for telephone (an association, not proof that the channel itself causes the difference).
- **Take the economic climate into account when timing campaigns.** Employment and economic indicators were the strongest signals.
- **Use model scores to help rank prospects** instead of treating everyone as equally likely to subscribe.

## 6. Limitations

The dataset represents Portuguese campaigns conducted in an earlier economic period, so the findings may not transfer directly to current markets or other countries. No external dataset or live campaign was used to test generalisability, operational costs or customer responses. Ordinal encoding may have imposed an artificial order on nominal categories, and correlated macroeconomic predictors complicate the interpretation of individual SHAP values. Finally, test-set comparisons were used in the reported model choice, which creates a possible post hoc selection bias.

## 7. Links
- 📄 [Full ML workflow with code and results (PDF)](notebooks/bank-marketing-term-deposit.pdf)
- 📘 [Dissertation report (PDF)](report/Dissertation_Report.pdf)
- 💼 [LinkedIn] https://www.linkedin.com/in/kamal-hassan-754298132/?isSelfProfile=true
