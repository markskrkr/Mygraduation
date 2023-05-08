# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:42:09 2023

@author: DELL
"""
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns 

sns.set_theme(style="whitegrid")
df2 = pd.read_csv('D:/毕业设计/faults1.csv')

X = df2.iloc[:,:-1]
y = df2.iloc[:, -1]

from imblearn.over_sampling import SMOTE

# 设置随机数种子
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X, y)

ss = StandardScaler()
data_ss = ss.fit_transform(X_smo)
df3 = pd.DataFrame(data_ss, columns=X_smo.columns)
df3["Label"] = y_smo
from sklearn.utils import shuffle
df3 = shuffle(df3)
X = df3.drop("Label",axis=1)
y = df3[["Label"]]

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)
params = { 
    'boosting_type': 'gbdt',  
    'objective': 'multiclass',  
    'num_class': 7,  
    'metric': 'multi_error',
    'num_leaves': 200,  
    'min_data_in_leaf': 20,  
    'learning_rate': 0.0033,
    'feature_fraction': 0.8,  
    'bagging_fraction': 0.8,  
    'bagging_freq': 7,  
    'lambda_l1': 0.4,  
    'lambda_l2': 0.4,  
    'min_gain_to_split': 0.2,  
    'verbose': -1,
    'num_threads':4,
    'max_depth':15,
    'eval_metric': 'multi_logloss'
}

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print('Training...')
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

print('Training...')
trn_data = lgb.Dataset(X_train_std, y_train)
val_data = lgb.Dataset(X_test_std, y_test, reference=trn_data)

num_rounds = 900
evals_result = {}
clf = lgb.train(params, 
                trn_data, 
                num_boost_round=num_rounds,
                valid_sets=[trn_data, val_data], 
                evals_result=evals_result,
                verbose_eval=100,
                early_stopping_rounds=100,)

print('Predicting...')
y_pred = clf.predict(X_test_std, num_iteration=clf.best_iteration)
y_pred = np.argmax(y_pred, axis=1)

result = classification_report(y_test, y_pred)


print(result)
# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Precision, Recall, F1 Score Bar Plot
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
n_classes = len(np.unique(y_test))
bar_width = 0.3
x = np.arange(n_classes)

plt.figure(figsize=(12, 6))
plt.bar(x, precision, width=bar_width, label='Precision')
plt.bar(x + bar_width, recall, width=bar_width, label='Recall')
plt.bar(x + 2 * bar_width, f1, width=bar_width, label='F1 Score')
plt.xticks(x + bar_width, np.arange(n_classes))
plt.xlabel('Class')
plt.ylabel('Score')
plt.legend()
plt.show()

# ROC Curve and AUC
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_pred_prob = clf.predict(X_test_std, num_iteration=clf.best_iteration)

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Training and Validation Curves
training_curve = evals_result['training']['multi_error']
validation_curve = evals_result['valid_1']['multi_error']

plt.figure(figsize=(10, 6))
plt.plot(training_curve, label="Training")
plt.plot(validation_curve, label="Validation")
plt.xlabel("Number of Boosting Rounds")
plt.ylabel("Error Rate")
plt.legend()
plt.show()

# Feature Importance
feature_names = X.columns.tolist()
importances = clf.feature_importance(importance_type='split')
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.subplots_adjust(left=0.3, bottom=0.2)
plt.show()