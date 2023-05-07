# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 21:30:47 2023

@author: DELL
"""


from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np



import seaborn as sns 

sns.set_theme(style="whitegrid")
df = pd.read_csv('D:/毕业设计/faults_origin.csv')

df1 = df.loc[:,"Pastry":]  # 7种不同的类型
df2 = df.loc[:,:"SigmoidOfAreas"]  # 全部是特征字段
columns = df1.columns.tolist()
for i in range(len(df1)):
    for col in columns:
        if df1.loc[i,col]==1:
            df1.loc[i,'label'] = col
dic = {}
for i, v in enumerate(columns):
    dic[v]=i  # 类别从0开始

df1['label'] = df1['label'].map(dic)   
df2['Label'] = df1['label']

X = df2.drop("Label",axis=1)
y = df2[["Label"]]

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
clf =OneVsRestClassifier(svm.SVC(C=1.0, kernel='rbf', degree=5, gamma='auto', coef0=1.0, 
                                 shrinking=True, probability=True, tol=0.001, cache_size=200, 
                                 class_weight=None, verbose=False, max_iter=-1,
                                 random_state=42))
clf.fit(X_train, y_train)
 

 

test_result = clf.predict(X_test)
result = classification_report(y_test, test_result,output_dict=True)

#df = pd.DataFrame(result).transpose()
#df.to_csv("C:/Users/DELL/result.csv", index= True)
clf.fit(X_train, y_train)

test_result = clf.predict(X_test)

# 混淆矩阵可视化
cm = confusion_matrix(y_test, test_result)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False, xticklabels=dic.keys(), yticklabels=dic.keys())
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for SVM')
plt.show()

# 支持向量数量可视化
support_vectors_counts = [len(clf.estimators_[i].support_vectors_) for i in range(len(clf.estimators_))]
plt.figure(figsize=(8, 6))
sns.barplot(x=list(dic.keys()), y=support_vectors_counts)
plt.xlabel('Class')
plt.ylabel('Number of Support Vectors')
plt.title('Number of Support Vectors for Each Class in SVM')
plt.show()