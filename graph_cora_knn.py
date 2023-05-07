import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

# Load your faulty-steel-plates dataset
data = pd.read_csv("D:/毕业设计/faults1.csv")
features = data.iloc[:, :-1]  # Extract features
labels = data.iloc[:, -1]  # Extract labels

# Create a k-NN graph
k = 5
nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(features)
distances, indices = nbrs.kneighbors(features)

# Create a NetworkX graph
G = nx.Graph()
for i, row in enumerate(indices):
    for neighbor in row:
        G.add_edge(i, neighbor)

# Assign features to nodes
for i, feat in enumerate(features.to_numpy()):
    G.nodes[i]["features"] = feat

# Convert labels to one-hot encoding
encoder = OneHotEncoder(sparse=False)
labels_one_hot = encoder.fit_transform(labels.to_numpy().reshape(-1, 1))

# Assign labels to nodes
for i, label in enumerate(labels_one_hot):
    G.nodes[i]["label"] = label

# Save the graph
nx.write_gpickle(G, "faulty_steel_plates_as_cora.gpickle")