# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:12:13 2023

@author: DELL
"""

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
import random
sns.set_theme(style="whitegrid")
df2 = pd.read_csv('D:/毕业设计/faults_smote.csv')

X = df2.iloc[:,:-1]
y = df2.iloc[:, -1]

def build_graph(sample):
    G = nx.Graph()


    for idx, (feature, value) in enumerate(sample.items()):
        G.add_node(idx, name=feature, feature=value)


    edges_group_1 = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
                     "SigmoidOfAreas","Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Luminosity_Index","Length_of_Conveyer",
                     "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness","Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
                     "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index"]

    for i in range(len(edges_group_1)):
        for j in range(i + 1, len(edges_group_1)):
            G.add_edge(sample.index.get_loc(edges_group_1[i]), sample.index.get_loc(edges_group_1[j]))

    return G
def build_graph_list(sample):
    G = nx.Graph()


    for idx, (feature, value) in enumerate(sample.items()):
        G.add_node(idx, name=feature, feature=value)


    edges_group_1 = ["X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas", "X_Perimeter", "Y_Perimeter",
                     "SigmoidOfAreas","Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Luminosity_Index","Length_of_Conveyer",
                     "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness","Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
                     "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index", "Orientation_Index"]

    for i in range(len(edges_group_1) - 1):
        G.add_edge(sample.index.get_loc(edges_group_1[i]), sample.index.get_loc(edges_group_1[i + 1]))

    return G


sample = X.iloc[0]

G = build_graph(sample)


#print(G.nodes(data=True))
print(G.edges)
print(G)


X = X.astype(float)
graphs = []
for i in range(len(X)):
    G = build_graph(X.iloc[i])
    graphs.append(G)

print(len(graphs))

import pickle
filename = 'graph_smote.pickle'

with open(filename, 'wb') as f:
    pickle.dump(graphs, f)
