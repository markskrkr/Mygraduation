# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 22:07:32 2023

@author: DELL
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pandas.plotting import parallel_coordinates
df = pd.read_csv('D:/毕业设计/faults1.csv')
X = df.iloc[:,:-7] 
Y = df.iloc[:, -1]
data = df.values

# 从每个类别中随机采样100个样本
sample_size = 10
sampled_data = pd.DataFrame()
for i in range(7):
    class_data = df.loc[df['Label'] == i, :]
    if len(class_data) >= sample_size:
        sampled_class_data = class_data.sample(n=sample_size, random_state=1)
    else:
        sampled_class_data = class_data.sample(n=sample_size, replace=True, random_state=1)
    sampled_data = pd.concat([sampled_data, sampled_class_data])

# 将采样后的数据进行标准化
sampled_data.iloc[:, :-1] = (sampled_data.iloc[:, :-1] - sampled_data.iloc[:, :-1].mean()) / sampled_data.iloc[:, :-1].std()

# 绘制平行坐标图
parallel_coordinates(sampled_data, 'Label', color=('#FF5733', '#C70039', '#900C3F', '#581845', '#3D9970', '#0074D9', '#F012BE'))
plt.title('Parallel Coordinates Plot')
plt.xlabel('Features')
plt.ylabel('Feature Values')
plt.tight_layout()
plt.show()
'''
max_values = data.max(axis=0)
min_values = data.min(axis=0)
normalized_data = (data - min_values) / (max_values - min_values)

# 绘制雷达图，将每个样本的所有特征值作为数据
n_features = X.shape[1]
angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
angles = np.concatenate((angles, [angles[0]]))
for i in range(normalized_data.shape[0]):
    values = normalized_data[i]
    values = np.concatenate((values, [values[0]]))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
plt.title('Radar Chart')
plt.grid(True)
plt.tight_layout()
plt.show()
'''