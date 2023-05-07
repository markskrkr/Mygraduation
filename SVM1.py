# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:25:14 2023

@author: DELL
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import sklearn.svm as svm
from sklearn.preprocessing import StandardScaler
data = pd.read_csv('D:/毕业设计/faults.csv')


X = data.iloc[:,:-7] 
Y = data.iloc[:, -1]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
stdsc = StandardScaler()
x_train = stdsc.fit_transform(x_train)
x_test = stdsc.transform(x_test)
print(">>> Load Dataset!")
def svc(kernel):
    return svm.SVC(kernel=kernel, decision_function_shape="ovo")


def nusvc():
    return svm.NuSVC(decision_function_shape="ovo")


def linearsvc():
    return svm.LinearSVC(multi_class="ovr")


def modelist():
    modelist = []
    kernalist = {"linear", "poly", "rbf", "sigmoid"}
    for each in kernalist:
        modelist.append(svc(each))
    modelist.append(nusvc())
    modelist.append(linearsvc())
    return modelist


def svc_model(model):
    print('==start to train model==')
    model.fit(x_train, y_train)
    acu_train = model.score(x_train, y_train)
    acu_test = model.score(x_test, y_test)
    y_pred = model.predict(x_test)
    recall = recall_score(y_test, y_pred, average="macro")
    print(acu_test)
    return acu_train, acu_test, recall


def run_svc_model(modelist):
    result = {"kernel": [],
              "acu_train": [],
              "acu_test": [],
              "recall": []
              }

    for model in modelist:
        acu_train, acu_test, recall = svc_model(model)
        try:
            result["kernel"].append(model.kernel)
        except:
            result["kernel"].append(None)
            print('wrong')
        result["acu_train"].append(acu_train)
        result["acu_test"].append(acu_test)
        result["recall"].append(recall)

    return pd.DataFrame(result)

run_svc_model(modelist())



