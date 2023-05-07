import pandas as pd
import networkx as nx
from sklearn.preprocessing import OneHotEncoder

# Load your faulty-steel-plates dataset as a DataFrame
data = pd.read_csv("D:/毕业设计/faults1.csv").sample(frac=1)
data = data.head(20)
features = data.iloc[:, :-1]  # Extract features
labels = data.iloc[:, -1]  # Extract labels

# Create a NetworkX graph
G = nx.Graph()

# Add nodes to the graph
for i in range(len(features)):
    G.add_node(i, features=features.iloc[i].to_numpy(), label=labels.iloc[i])

# Add edges between nodes with the same label
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        if labels.iloc[i] == labels.iloc[j]:
            G.add_edge(i, j)

# Convert labels to one-hot encoding
encoder = OneHotEncoder(sparse=False)
labels_one_hot = encoder.fit_transform(labels.to_numpy().reshape(-1, 1))

# Update the node labels with one-hot encoding
for i, label in enumerate(labels_one_hot):
    G.nodes[i]["label"] = label

# Save the graph
nx.write_gpickle(G, "faulty_steel_plates_as_cora_50.pickle")