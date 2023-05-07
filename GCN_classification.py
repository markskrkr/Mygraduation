import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import pickle
import pandas as pd
import torch.nn.functional as F
import random
from torch_geometric.data import DataLoader, Data, Batch
import networkx as nx

device = torch.device('cpu')

# Load labels
df2 = pd.read_csv('D:/毕业设计/faults1.csv')
labels = df2.iloc[:, -1]

# Load graphs
with open("D:/毕业设计/Scripts/graph_list.pickle", "rb") as f:
    graph_list = pickle.load(f)

def networkx_to_data(G):
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
    x = torch.tensor(list(nx.get_node_attributes(G, "feature").values()), dtype=torch.float).unsqueeze(-1)
    return Data(x=x, edge_index=edge_index)

# Convert networkx graphs to PyTorch Geometric Data objects and assign labels
data_list = [networkx_to_data(G) for G in graph_list]
for i, data in enumerate(data_list):
    data.y = torch.tensor([labels[i]], dtype=torch.long)

# Split the dataset
random.shuffle(data_list)
train_dataset = data_list[:int(len(data_list) * 0.8)]
test_dataset = data_list[int(len(data_list) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels=16, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, data.batch)
        return F.log_softmax(x, dim=1)

model = GCN(num_features=1, num_classes=7).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def My_test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)

for epoch in range(200):
    loss = train()
    train_acc = My_test(train_loader)
    test_acc = My_test(test_loader)
    print(f"Epoch: {epoch+1}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")