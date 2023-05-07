# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:47:39 2023

@author: DELL
"""
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report


data = pd.read_csv('D:/毕业设计/faults1.csv')


X = data.iloc[:,:-7] 
Y = data.iloc[:, -1]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)

stdsc = StandardScaler()
X_train = stdsc.fit_transform(X_train)
X_test = stdsc.transform(X_test)

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


def CreateBalancedSampleWeights(y_train, largest_class_weight_coef):
    classes = np.unique(y_train, axis = 0)
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key : value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]] * largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]
    return sample_weights

largest_class_weight_coef = max(Y.value_counts().values)/Y.shape[0]
Myweight = CreateBalancedSampleWeights(Y_train, largest_class_weight_coef)

data_dmatrix = xgb.DMatrix(data=X_train,label=Y_train,weight = Myweight)
test_dmatrix = xgb.DMatrix(data=X_test,label=Y_test)
dtest=xgb.DMatrix(X_test)
num_rounds = 1800
model = xgb.train(params, data_dmatrix, 
                  num_rounds,evals=[(data_dmatrix,'train'),(test_dmatrix,'vaild')],
                  
                  )

preds = model.predict(test_dmatrix)

result = classification_report(Y_test, preds)
print(result)
'''
predictions=[round(value) for value in preds]

accuracy=accuracy_score(Y_test,predictions)
print('accu: %.2f%%' % (accuracy*100))
'''
