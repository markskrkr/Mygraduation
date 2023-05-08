# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 17:03:29 2023

@author: DELL
"""

import pandas as pd


precision = [0.90, 0.85, 0.95, 0.80, 0.75, 0.70, 0.85, 0.82, 0.83]
recall = [0.85, 0.80, 0.92, 0.75, 0.70, 0.68, 0.80, 0.78, 0.79]
f1_score = [0.87, 0.82, 0.94, 0.77, 0.72, 0.68, 0.83, 0.80, 0.81]
support = [100, 200, 150, 300, 250, 200, 150, 1450, 3000]


index = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults', 'macro average', 'weighted average']


columns = ['precision', 'recall', 'f1-score', 'support']


data = [precision, recall, f1_score, support]
data = [data[i] + [sum(data[i])] for i in range(3)]
data = data + [support]


df = pd.DataFrame(data=data, columns=columns, index=index)
df.loc['accuracy'] = ['', '', '', '0.85']


print(df)