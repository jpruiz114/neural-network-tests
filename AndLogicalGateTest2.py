import torch
import torch.nn as nn

# Data (AND truth table)
X = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]])
y = torch.tensor([[0.],[0.],[0.],[1.]])

# Model: a single linear unit (no hidden layer)
model = nn.Sequential(nn.Linear(2, 1))
loss_fn = nn.BCEWithLogitsLoss()     # expects raw logits
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2000):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

# Inference: apply sigmoid to logits
with torch.no_grad():
    probs = torch.sigmoid(model(X))
    preds = (probs > 0.5).float()
print("AND probs:\n", probs.round(decimals=3))
print("AND preds:\n", preds)
