# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 21:55:09 2023

@author: DELL
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 加载数据
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
pca = PCA(n_components="mle")
pca_data = pca.fit_transform(df3)
X = df3.drop("Label",axis=1)
y = df3[["Label"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练XGBoost模型
xgb_model = XGBClassifier(n_estimators=900, learning_rate=0.0025, random_state=42, max_depth=15, subsample=0.8, colsample_bytree=0.8, gamma=0.1)
xgb_model.fit(X_train, y_train)

# 训练LightGBM模型
lgbm_model = LGBMClassifier(n_estimators=900, learning_rate=0.003, random_state=42, max_depth=15, subsample=0.8, colsample_bytree=0.8, min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1)
lgbm_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_preds)
print("XGBoost accuracy: {:.2f}%".format(xgb_accuracy * 100))

# 计算LightGBM准确率
lgbm_preds = lgbm_model.predict(X_test)
lgbm_accuracy = accuracy_score(y_test, lgbm_preds)
print("LightGBM accuracy: {:.2f}%".format(lgbm_accuracy * 100))
# 获取XGBoost和LightGBM的预测结果
xgb_preds_train = xgb_model.predict_proba(X_train)
lgbm_preds_train = lgbm_model.predict_proba(X_train)

xgb_preds_test = xgb_model.predict_proba(X_test)
lgbm_preds_test = lgbm_model.predict_proba(X_test)

# 合并XGBoost和LightGBM的预测结果
ensemble_train = np.hstack((xgb_preds_train, lgbm_preds_train))
ensemble_test = np.hstack((xgb_preds_test, lgbm_preds_test))
X_train_ensemble, X_val_ensemble, y_train_ensemble, y_val_ensemble = train_test_split(ensemble_train, y_train, test_size=0.2, random_state=42)
# 训练CatBoost模型
catboost_model = CatBoostClassifier(iterations=120, learning_rate=0.004, random_seed=42, verbose=0, custom_loss='Accuracy')
catboost_model.fit(X_train_ensemble, y_train_ensemble, eval_set=(X_val_ensemble, y_val_ensemble))

# 使用CatBoost融合模型进行预测
ensemble_preds = catboost_model.predict(ensemble_test)
print(ensemble_test[0])
'''
# 计算准确率
ensemble_accuracy = accuracy_score(y_test, ensemble_preds)
print("Ensemble accuracy: {:.2f}%".format(ensemble_accuracy * 100))
models = ['XGBoost', 'LightGBM', 'Ensemble']
accuracies = [xgb_accuracy, lgbm_accuracy, ensemble_accuracy]

plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.show()

# 融合模型的混淆矩阵
ensemble_cm = confusion_matrix(y_test, ensemble_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(ensemble_cm, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Ensemble Model Confusion Matrix')
plt.show()

# 训练和验证损失曲线
# 需要注意的是，我们无法直接从CatBoostClassifier获取损失值，但可以查看每次迭代的训练和验证精度。
train_accuracy = catboost_model.get_evals_result()['learn']['Accuracy']
val_accuracy = catboost_model.get_evals_result()['validation']['Accuracy']

plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Curve')
plt.legend()
plt.show()
'''