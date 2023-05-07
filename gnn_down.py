import pandas as pd
import seaborn as sns
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

sns.set_theme(style="whitegrid")
df2 = pd.read_csv('D:/毕业设计/faults1.csv')

X = df2.iloc[:, :-1]
y = df2.iloc[:, -1]

def build_fully_connected_graph(sample):
    G = nx.Graph()

    # 添加节点
    for idx, (feature, value) in enumerate(sample.items()):
        G.add_node(idx, name=feature, feature=value)

    # 添加边（全连接）
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            G.add_edge(i, j)

    return G

def create_pyg_data(G):
    node_features = torch.tensor([data['feature'] for _, data in G.nodes(data=True)], dtype=torch.float).unsqueeze(1)
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()

    return Data(x=node_features, edge_index=edge_index)

# 用一个样本构建全连接图
sample = X.iloc[0]
G = build_fully_connected_graph(sample)

# 构建PyTorch Geometric数据
pyg_data = create_pyg_data(G)

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 将X_train转换为图结构的数据
graphs = [create_pyg_data(build_fully_connected_graph(sample)) for _, sample in X_train.iterrows()]

# 训练GNN模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(pyg_data.num_node_features, len(y.unique())).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    total_loss = 0
    for graph in graphs:
        graph = graph.to(device)
        optimizer.zero_grad()
        out = model(graph)
        loss = F.nll_loss(out, torch.tensor([y_train.loc[graph.y]], dtype=torch.long).to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {total_loss / len(graphs)}')