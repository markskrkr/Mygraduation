import torch
import numpy as np
import pandas as pd
import pickle
import random
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import from_networkx
from torch_geometric.data import DataLoader
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
device = torch.device('cpu')

df2 = pd.read_csv('D:/毕业设计/faults_smote.csv')
labels = df2.iloc[:, -1]

def load_graph(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    graph_list = []
    for item in data:
        graph_list.append(item)

    return graph_list

def networkx_to_pyg(graph):
    features = torch.tensor([features["feature"] for _, features in graph.nodes(data=True)], dtype=torch.float).view(-1, 1)
    pyg_graph = from_networkx(graph)
    pyg_graph.x = features
    return pyg_graph

graph_list = load_graph("D:/毕业设计/Scripts/graph_smote.pickle")
pyg_graph_list = [networkx_to_pyg(graph) for graph in graph_list]

for i, pyg_graph in enumerate(pyg_graph_list):
    pyg_graph.y = torch.tensor([labels[i]], dtype=torch.long)

random.shuffle(pyg_graph_list)
train_dataset = pyg_graph_list[:int(len(pyg_graph_list) * 0.8)]
test_dataset = pyg_graph_list[int(len(pyg_graph_list) * 0.8):]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

    def extract_embeddings(self, x, edge_index):
        act = getattr(F, self.activation)
        for i, conv in enumerate(self.convs[:-1]):
            x = act(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

def extract_embeddings(loader):
    model.eval()
    embeddings = []
    labels = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        out = global_mean_pool(out, data.batch)
        embeddings.append(out.cpu().detach().numpy())
        labels.append(data.y.cpu().detach().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.hstack(labels)
    return embeddings, labels


model = GAT(num_features=1, num_classes=7, num_layers=2, num_heads=4, hidden_dim=8, dropout=0.6).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)


num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = global_mean_pool(out, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    train_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch + 1}, Loss: {train_loss:.4f}')


train_embeddings, train_labels = extract_embeddings(train_loader)
test_embeddings, test_labels = extract_embeddings(test_loader)

print((len(train_embeddings)))



xgb_classifier = xgb.XGBClassifier(n_estimators=500, max_depth=15, min_child_weight=6, reg_alpha=1, reg_lambda=1,learning_rate=0.02)
eval_set = [(train_embeddings, train_labels), (test_embeddings, test_labels)]
xgb_classifier.fit(train_embeddings, train_labels, eval_set=eval_set, eval_metric='merror', verbose=True)

train_preds = xgb_classifier.predict(train_embeddings)
test_preds = xgb_classifier.predict(test_embeddings)

train_acc = accuracy_score(train_labels, train_preds)
test_acc = accuracy_score(test_labels, test_preds)

print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
import matplotlib.pyplot as plt


train_accuracy = [1 - x for x in xgb_classifier.evals_result_['validation_0']['merror']]
test_accuracy = [1 - x for x in xgb_classifier.evals_result_['validation_1']['merror']]
epochs = range(len(train_accuracy))


plt.plot(epochs, train_accuracy, label='Train')
plt.plot(epochs, test_accuracy, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy')
plt.legend()
plt.grid(True)
plt.show()
'''
train_embeddings, train_labels = extract_embeddings(train_loader)
test_embeddings, test_labels = extract_embeddings(test_loader)

# 使用SMOTE进行样本均衡处理
smote = SMOTE()
train_embeddings_resampled, train_labels_resampled = smote.fit_resample(train_embeddings, train_labels)

# 准备xgboost训练所需的数据格式
dtrain = xgb.DMatrix(train_embeddings_resampled, label=train_labels_resampled)
dtest = xgb.DMatrix(test_embeddings, label=test_labels)

# 设置XGBoost的参数
params = {
    'objective': 'multi:softmax',
    'num_class': 7,
    'eval_metric': 'merror'
}

# 使用xgb.cv函数进行交叉验证
cv_results = xgb.cv(params, dtrain, num_boost_round=100, nfold=5, seed=42, verbose_eval=10)

# 训练xgboost分类器
xgb_classifier = xgb.train(params, dtrain, num_boost_round=cv_results.shape[0])

# 使用训练好的模型进行预测
train_preds = xgb_classifier.predict(dtrain)
test_preds = xgb_classifier.predict(dtest)

train_acc = accuracy_score(train_labels_resampled, train_preds)
test_acc = accuracy_score(test_labels, test_preds)

print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

print(f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
'''