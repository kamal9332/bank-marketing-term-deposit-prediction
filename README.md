# Predicting term-deposit subscriptions using ML on 41,176 bank marketing records

A leakage-controlled machine learning pipeline comparing Logistic Regression, Random Forest and XGBoost, with SHAP used to explain what drives subscriptions.

## 1. Business Problem
Telemarketing campaigns convert few customers, so many calls are wasted. The bank needs to target better: contact the right customers, through the right channel, at the right time.

## 2. Dataset
- Source: UCI Portuguese Bank Marketing dataset (Moro et al., 2014)
- 41,188 rows → **41,176 after cleaning** (12 duplicates removed)
- **19 predictors**: customer, campaign and macroeconomic features
- Target: subscribed to a term deposit (yes/no). **Imbalanced**: only about 11.27% subscribed.

## 3. Method
Framework: **CRISP-DM**
- Dropped `duration` to prevent data leakage
- Engineered `previously_contacted` from `pdays`
- Stratified 60/20/20 train/validation/test split
- Preprocessing fitted on training data only (one-hot encoding and scaling for LR, ordinal encoding for RF and XGBoost)
- SMOTE applied inside the CV folds only
- 5-fold CV and GridSearchCV for tuning

## 4. Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | **0.8817** | **0.4782** | 0.5442 | **0.5091** | 0.7967 |
| Random Forest | 0.8798 | 0.4708 | 0.5377 | 0.5020 | **0.7977** |
| Logistic Regression | 0.8299 | 0.3590 | **0.6487** | 0.4622 | 0.7965 |

XGBoost had the best F1 (the primary metric for imbalanced data). Logistic Regression found the most subscribers (highest recall) but made more wasted calls. ROC-AUC was about 0.80 for all three models.

## 5. Key Insights and Business Recommendations
Top SHAP drivers: `nr.employed` (0.447), `contact` (0.229), `emp.var.rate` (0.154), `poutcome` (0.148).

- **Prioritise customers with a successful previous campaign.** They subscribed at 65.11%, against 8.83% for customers with no previous contact.
- **Use cellular contact.** It converted at 14.74%, against 5.23% for landline.
- **Time campaigns to the economic climate.** Employment and economic indicators were the strongest signals.
- **Rank prospects by model score** instead of treating everyone as equally likely to subscribe.

## 6. Limitations
- Single, historical Portuguese dataset, so external validation is needed.
- No live testing, so the model should support decisions, not make them automatically.
- The economic features are strongly correlated, so their effects are not independent causes.
- Model differences are small (F1 0.46 to 0.51). Model selection also looked at test results, which can bias the choice.

## 7. How to Run
```bash
git clone https://github.com/YOUR-USERNAME/bank-marketing-term-deposit-prediction.git
cd bank-marketing-term-deposit-prediction
pip install -r requirements.txt
jupyter notebook notebooks/bank-marketing-term-deposit.ipynb
```
Then run all cells. The dataset is already in `/data`.

## 8. Links
- 📄 [Full ML workflow with code and results (PDF)](notebooks/bank-marketing-term-deposit.pdf)
- 📘 [Dissertation report (PDF)](report/Dissertation_Report.pdf)
- 💼 [LinkedIn] https://www.linkedin.com/in/kamal-hassan-754298132/?isSelfProfile=true
