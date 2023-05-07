# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:23:32 2023

@author: DELL
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

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



nb = GaussianNB(var_smoothing=1e-9)

# 训练朴素贝叶斯分y类器
nb.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = nb.predict(X_test)

# 计算分类准确率
result = classification_report(y_test,y_pred,output_dict=True)
df = pd.DataFrame.from_dict(classification_report(y_test, y_pred, output_dict=True)).round(2)
df_transposed = df.transpose()
df_transposed.to_csv("D:/result_Naive.csv", index=True)
# 添加可视化部分
# 1. 条件概率表（CPT）可视化
# 注意：因为我们使用的是GaussianNB，所以条件概率表不能直接获得，但我们可以使用均值和方差
# 来构建一个近似的CPT。
means = nb.theta_
variances = nb.sigma_
n_classes = len(np.unique(y))

fig, ax = plt.subplots(figsize=(12, 6))
for class_idx in range(n_classes):
    sns.kdeplot(X_test.dot(means[class_idx, :]), ax=ax, label=f'Class {class_idx + 1}')

ax.set_xlabel('Feature Combination')
ax.set_ylabel('Probability Density')
ax.set_title('Estimated Conditional Probability Distributions')
ax.legend()
plt.show()

# 2. 敏感性分析
# 改变一个特征的值，查看对预测概率的影响
selected_feature_index = 0  # 选择一个特征进行敏感性分析
min_value, max_value = X.iloc[:, selected_feature_index].min(), X.iloc[:, selected_feature_index].max()
step = (max_value - min_value) / 100
feature_values = np.arange(min_value, max_value, step)
prob_changes = []

for val in feature_values:
    X_test_new = X_test.copy()
    X_test_new.iloc[:, selected_feature_index] = val
    y_proba_new = nb.predict_proba(X_test_new)
    prob_changes.append(y_proba_new.mean(axis=0))

prob_changes = np.array(prob_changes)

for class_idx in range(n_classes):
    plt.plot(feature_values, prob_changes[:, class_idx], label=f'Class {class_idx + 1}')

plt.xlabel(f'Feature {selected_feature_index + 1} Value')
plt.ylabel('Predicted Probability')
plt.title('Sensitivity Analysis')
plt.legend()
plt.show()