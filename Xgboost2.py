# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:04:05 2023

@author: DELL
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support


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
pca = PCA(n_components="mle")
pca_data = pca.fit_transform(df3)
X = df3.drop("Label",axis=1)
y = df3[["Label"]]

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=4)

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 7,
    'gamma': 0.2,
    'max_depth': 8,
    'lambda': 2,
    'subsample': 0.9,
    'colsample_bytree': 0.5,
    'eta': 0.002,
    'seed': 42,
    'nthread': 1,
    'min_child_weight':0.5,
    'eval_metric':'auc',
    'colsample_bylevel': 0.5,  'tree_method': 'hist'
}
print('start')




data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
test_dmatrix = xgb.DMatrix(data=X_test,label=y_test)
dtest=xgb.DMatrix(X_test)
evals_result = {}#
num_rounds = 600
model = xgb.train(params, data_dmatrix, 
                  num_rounds,evals=[(data_dmatrix,'train'),(test_dmatrix,'vaild')],
                  evals_result=evals_result,
                  )

preds = model.predict(test_dmatrix)


report = classification_report(y_test, preds)
print(report)
model.save_model('D:/毕业设计/xgboost_model.json')
report = classification_report(y_test, preds)
df = pd.DataFrame.from_dict(classification_report(y_test, preds, output_dict=True)).round(2)
df_transposed = df.transpose()
df_transposed.to_csv("D:/result_X.csv", index=True)

# 添加可视化部分
# 1. 混淆矩阵热力图
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# 2. 精确率、召回率和 F1 分数
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, preds, average=None)
categories = list(range(1, 8))  # 根据你的实际类别数进行调整
plt.bar(categories, precision, alpha=0.6, label='Precision')
plt.bar(categories, recall, alpha=0.6, label='Recall')
plt.bar(categories, f1_score, alpha=0.6, label='F1 Score')
plt.xlabel('Category')
plt.ylabel('Score')
plt.legend()
plt.show()

# 3. ROC 曲线和 AUC
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)
categories = np.unique(y)
y_test_bin = label_binarize(y_test, classes=categories)

# 获取模型预测概率
y_pred_margin = model.predict(test_dmatrix, output_margin=True)
y_pred_proba = softmax(y_pred_margin)

# 计算每个类别的ROC曲线和AUC
fpr, tpr, roc_auc = {}, {}, {}
for i in range(len(categories)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制ROC曲线
for i in range(len(categories)):
    plt.plot(fpr[i], tpr[i], label=f'Category {i + 1} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# 4. 算法训练和验证曲线
train_auc = evals_result['train']['auc']
valid_auc = evals_result['vaild']['auc']
plt.plot(range(len(train_auc)), train_auc, label='Train')
plt.plot(range(len(valid_auc)), valid_auc, label='Validation')
plt.xlabel('Number of Rounds')
plt.ylabel('AUC')
plt.legend()
plt.show()

# 5. 特征重要性
xgb.plot_importance(model)
plt.subplots_adjust(left=0.3, bottom=0.2)
plt.show()