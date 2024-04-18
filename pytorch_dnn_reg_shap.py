!pip install torch shap


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shap

# ダミーデータの生成
np.random.seed(0)
X_train = np.random.rand(100, 5)
y_train = np.random.rand(100)

# PyTorchでのDNNの定義
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(5, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルの定義と学習
model = DNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(torch.Tensor(X_train))
    loss = criterion(outputs.squeeze(), torch.Tensor(y_train))
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{100}], Loss: {loss.item():.4f}')

# DeepExplainerで説明性を計算
explainer = shap.DeepExplainer(model, torch.Tensor(X_train))
shap_values = explainer.shap_values(torch.Tensor(X_train))

# SHAP summary plotの表示
shap.summary_plot(shap_values, X_train)
