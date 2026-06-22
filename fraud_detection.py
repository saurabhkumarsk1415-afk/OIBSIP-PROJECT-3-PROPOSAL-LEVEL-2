"""
Credit Card Fraud Detection
Project 3 Proposal - Level 2
Dataset: creditcard.csv — 284,807 real European credit card transactions (Sept 2013)
Source: Kaggle / Machine Learning Group - ULB
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    average_precision_score)
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# 1. LOAD & INSPECT
# ══════════════════════════════════════════════════════════
print("Loading creditcard.csv...")
df = pd.read_csv('creditcard.csv')
print(f"Shape: {df.shape}")
print(f"Nulls: {df.isnull().sum().sum()}")
print(f"Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")
print(f"Legitimate: {(df['Class']==0).sum()}")

# ══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════
# V1-V28 are already PCA-scaled; standardize Amount and Time
df['scaled_amount'] = StandardScaler().fit_transform(df[['Amount']])
df['scaled_time']   = StandardScaler().fit_transform(df[['Time']])
df.drop(['Amount', 'Time'], axis=1, inplace=True)

# ══════════════════════════════════════════════════════════
# 3. TRAIN-TEST SPLIT
# ══════════════════════════════════════════════════════════
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")
print(f"Test fraud count: {y_test.sum()}")

# ══════════════════════════════════════════════════════════
# 4. HANDLE CLASS IMBALANCE (training set only)
# ══════════════════════════════════════════════════════════
rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
print(f"After undersampling: {pd.Series(y_train_res).value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════
# 5. TRAIN MODELS
# ══════════════════════════════════════════════════════════
print("\nTraining models...")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, C=0.01, random_state=42)
lr.fit(X_train_res, y_train_res)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
dt.fit(X_train_res, y_train_res)

# Neural Network (MLP)
nn = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100,
                   early_stopping=True, random_state=42)
nn.fit(X_train_res, y_train_res)

# ══════════════════════════════════════════════════════════
# 6. EVALUATE
# ══════════════════════════════════════════════════════════
print("\n=== MODEL EVALUATION ===")
for name, model in [('Logistic Regression', lr), ('Decision Tree', dt), ('Neural Network', nn)]:
    pred  = model.predict(X_test)
    proba = model.predict_proba(X_test)[:,1]
    print(f"\n{name}")
    print(f"  Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_test, pred, zero_division=0):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, proba):.4f}")
    print(f"  PR-AUC:    {average_precision_score(y_test, proba):.4f}")
    print(classification_report(y_test, pred, target_names=['Legitimate','Fraudulent'], zero_division=0))
