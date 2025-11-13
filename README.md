# Credit Card Fraud Detection

## Introduction
This project aims to detect fraudulent credit card transactions in a highly imbalanced dataset.
Dataset taken from: [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection).

Key objectives:
- Build a preprocessing and feature engineering pipeline.  
- Address the class imbalance problem.  
- Compare multiple machine learning models (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM).  
- Evaluate models using metrics suitable for imbalanced data (ROC-AUC, PR-AUC).  

## Pipeline
- **Preprocessing & Feature Engineering**  
  - Time features: extracted year, month, day, hour, minute; created `time_of_day` and `time_delta`.  
  - Demographic features: calculated customer age and generational cohorts.  
  - Address features: split street into components, normalized ZIP codes, extracted ZIP-prefix, calculated haversine distance between customer and merchant as `distance_km`.  
  - Occupation: simplified to the primary job label.  
  - Encoding:  
    - High-cardinality → Frequency Encoding.  
    - Low-cardinality → One-Hot Encoding.  
    - Generations → Label Encoding.  

- **Class Imbalance Handling**  
  - Applied undersampling to reduce majority class size.  
  - Applied SMOTE oversampling to generate synthetic fraud cases.  
  - Hybrid resampling to balance the dataset.  

- **Baseline Benchmark**  
  - Logistic Regression (with `class_weight='balanced'`) trained as the baseline model.  

- **Modeling & Evaluation**  
  - Models compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM.  
  - Metrics: Precision, Recall, F1, ROC-AUC and PR-AUC (preferred over accuracy for imbalanced data).  

## Key Results
- Logistic Regression baseline showed moderate but stable performance, indicating the dataframe has enough information for the model to distinguish between non-fraud and fraud.  
- Resampling improved recall and fraud detection ability.  
- Ensemble models (Random Forest, LightGBM, XGBoost) achieved higher ROC-AUC and PR-AUC than Logistic Regression.  

## Comparision table:
| Model          | Precision (0) | Recall (0) | Precision (1) | Recall (1) | ROC-AUC | PR-AUC |
|----------------|---------------|------------|---------------|------------|---------|--------|
| LogisticReg    | 0.999         | 0.931      | 0.021         | 0.701      | 0.894   | 0.155  |
| RandomForest   | 0.999         | 0.999      | 0.720         | 0.817      | 0.993   | 0.837  |
| XGBoost        | 0.999         | 0.999      | 0.81          | 0.81       | 0.997   | 0.873  |
| LightGBM Auto	 |	1.0					 | 0.995      | 0.438         | 0.924      | 0.998	 | 0.888  |
| LightGBM Manual|	1.0	         | 0.998	    | 0.593	        | 0.900	     | 0.998	 | 0.892  |

