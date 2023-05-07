# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:33:38 2023

@author: DELL
"""

import pandas as pd
import numpy as np

import seaborn as sns 
from sklearn.model_selection import train_test_split
sns.set_theme(style="whitegrid")
df = pd.read_csv('D:/毕业设计/faults_origin.csv')

df1 = df.loc[:,"Pastry":]  # 7种不同的类型
df2 = df.loc[:,:"SigmoidOfAreas"]  # 全部是特征字段
columns = df1.columns.tolist()
for i in range(len(df1)):
    for col in columns:
        if df1.loc[i,col]==1:
            df1.loc[i,'label'] = col
print(df1['label'])
dic = {}
for i, v in enumerate(columns):
    dic[v]=i  # 类别从0开始

df1['label'] = df1['label'].map(dic)   
df2['Label'] = df1['label']

X = df2.drop("Label",axis=1)
y = df2[["Label"]]
print(df2.head())

from imblearn.over_sampling import SMOTE

# 设置随机数种子
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X, y)
from sklearn.preprocessing import StandardScaler


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
                                                    random_state=4)
from sklearn.model_selection import cross_val_score  # 交叉验证得分
from sklearn import metrics  # 模型评价


def build_model(model, X_test, y_test):
    
    model.fit(X_train, y_train)
    # 预测概率
    y_proba = model_LR.predict_proba(X_test)
    # 找出概率值最大的所在索引，作为预测的分类结果
    y_pred = np.argmax(y_proba,axis=1)
    y_test = np.array(y_test).reshape(943)
    
    print(f"{model}模型得分：")
    print("召回率: ",metrics.recall_score(y_test, y_pred, average="macro"))
    print("精准率: ",metrics.precision_score(y_test, y_pred, average="macro"))
from sklearn.linear_model import LogisticRegression  
# 建立模型
model_LR = LogisticRegression()
# 调用函数
build_model(model_LR, X_test, y_test)
'''
parameters = X.columns[:-1].tolist()
# 两个基本参数：设置行、列

fig = make_subplots(rows=7, cols=4)  # 1行2列

for i, v in enumerate(parameters):  
    r = i // 4 + 1
    c = (i+1) % 4 
    
    if c ==0:
        fig.add_trace(go.Box(y=X[v].tolist(),name=v),
                 row=r, col=4)
    else:
        fig.add_trace(go.Box(y=X[v].tolist(),name=v),
                 row=r, col=c)
    
fig.update_layout(width=1000, height=900)
fig.write_image('D:/毕业设计/subplots_1.png', scale=10)
fig.show()
'''
#smote 数据平衡
