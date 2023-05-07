# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:30:01 2023

@author: DELL
"""

import pandas as pd

import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import xgboost as xgb
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

sns.set_theme(style="whitegrid")
df2 = pd.read_csv('D:/毕业设计/gan_faults.csv')

X_train = df2.iloc[:,:-7] 
y_train = df2.iloc[:, -1]

data = pd.read_csv('D:/毕业设计/faults1.csv')


X_test = data.iloc[:,:-7] 
y_test = data.iloc[:, -1]




params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 7,               # 类别数，与 multisoftmax 并用
    'gamma': 0.2,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 20,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.9,              # 随机采样训练样本
    'colsample_bytree': 0.8,       # 生成树时进行的列采样
    'eta': 0.002,                  # 如同学习率
    'seed': 42,
    'nthread': -1,                  # cpu 线程数
    'min_child_weight':0.5,
    'eval_metric':'auc',
}
print('start')




data_dmatrix = xgb.DMatrix(data=X_train,label=y_train)
test_dmatrix = xgb.DMatrix(data=X_test,label=y_test)
dtest=xgb.DMatrix(X_test)
evals_result = {}#
num_rounds = 750
model = xgb.train(params, data_dmatrix, num_rounds,evals=[(data_dmatrix,'train'),(test_dmatrix,'vaild')],
evals_result=evals_result,
                  )

preds = model.predict(test_dmatrix)


report = classification_report(y_test, preds)
rows = report.strip().split('\n')
data = []
for row in rows[2:-3]:
    row_data = row.split()
    data.append(row_data[1:])

# 添加平均值指标
macro_data = rows[-3].split()[1:]
weighted_data = rows[-2].split()[1:]
data.append(macro_data)
data.append(weighted_data)

# 创建表格
df = pd.DataFrame(data, columns=['precision', 'recall', 'f1-score', 'support'], index=rows[1].split()[1:])

# 显示表格
print(df)

train_loss=list(evals_result['train'].values())[0]
valid_loss=list(evals_result['vaild'].values())[0]
x_scale=[i for i in range(len(train_loss))]
plt.figure(figsize=(10,10))
plt.title('loss')
plt.plot(x_scale,train_loss,label='train',color='r')
plt.plot(x_scale,valid_loss,label='vaild',color='b')
plt.legend()
plt.show()
