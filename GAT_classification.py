import torch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import from_networkx
import pickle
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import random
import numpy as np
device = torch.device('cpu')
df2 = pd.read_csv('D:/毕业设计/faults1.csv')
labels = df2.iloc[:, -1]
def load_graph(filename):

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    graph_list = []
    for item in data:
        graph_list.append(item)

    return graph_list

def networkx_to_pyg(graph):
    # 为每个节点提取特征并将它们转换为torch.Tensor
    features = torch.tensor([features["feature"] for _, features in graph.nodes(data=True)], dtype=torch.float).view(-1, 1)

    # 从networkx图转换为PyTorch Geometric图
    pyg_graph = from_networkx(graph)

    # 为PyTorch Geometric图分配节点特征
    pyg_graph.x = features

    return pyg_graph

graph_list = load_graph("D:/毕业设计/Scripts/graph_list.pickle")
# 假设loaded_graph_list包含1941个图
pyg_graph_list = [networkx_to_pyg(graph) for graph in graph_list]

#labels标签
for i, pyg_graph in enumerate(pyg_graph_list):
    pyg_graph.y = torch.tensor([labels[i]], dtype=torch.long)

from torch_geometric.data import DataLoader
random.shuffle(pyg_graph_list)
# 将数据集划分为训练集和测试集
train_dataset = pyg_graph_list[:int(len(pyg_graph_list) * 0.8)]
test_dataset = pyg_graph_list[int(len(pyg_graph_list) * 0.8):]

# 创建数据加载器


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_acc_list = []
test_acc_list = []
class GAT(torch.nn.Module):
    def __init__(self, num_features, num_classes, num_layers=2, num_heads=8, hidden_dim=8, dropout=0.6, activation='relu'):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_features, hidden_dim, heads=num_heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads))
        self.convs.append(GATConv(hidden_dim * num_heads, num_classes, heads=1))

    def forward(self, x, edge_index):
        act = getattr(F, self.activation)
        for i, conv in enumerate(self.convs[:-1]):
            x = act(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)
learning_rate = 0.1
weight_decay = 5e-4
num_epochs = 50
train_batch_size = 32
test_batch_size = 32
num_heads = 4
hidden_dim = 8
dropout = 0.6
num_layers = 2
lr_scheduler_step_size = 30
lr_scheduler_gamma = 0.5
# 初始化模型
model = GAT(num_features=1, num_classes=7, num_layers=num_layers, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 训练和测试函数
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = global_mean_pool(out, data.batch)  # 添加汇集函数
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
        out = model(data.x, data.edge_index)
        out = global_mean_pool(out, data.batch)  # 添加汇集函数
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
    return correct / len(loader.dataset)
# 训练模型并测试准确率
for epoch in range(num_epochs):
    train_loss = train()
    train_acc = My_test(train_loader)
    test_acc = My_test(test_loader)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f'Epoch: {epoch+1}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


import matplotlib.pyplot as plt

plt.plot(train_acc_list, label='Train Acc')
plt.plot(test_acc_list, label='Test Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.1, 1)
plt.title('Train and Test Accuracy')
plt.legend()

plt.show()
