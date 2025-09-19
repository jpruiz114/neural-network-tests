import torch
import torch.nn as nn

# Data (NOT truth table)
X = torch.tensor([[0.],
                  [1.]])
y = torch.tensor([[1.],
                  [0.]])

# Model: Single linear layer (NOT is linearly separable)
model = nn.Sequential(nn.Linear(1, 1))
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2000):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    probs = torch.sigmoid(model(X))
    preds = (probs > 0.5).float()
print("NOT probs:\n", probs.round(decimals=3))
print("NOT preds:\n", preds)
