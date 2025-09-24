import torch
import torch.nn as nn

# Data (XNOR truth table)
X = torch.tensor([[0.,0.],
                  [0.,1.],
                  [1.,0.],
                  [1.,1.]])
y = torch.tensor([[1.],[0.],[0.],[1.]])

# Model: 2 -> 4 -> 1 with ReLU
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(5000):
    logits = model(X)
    loss = loss_fn(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

with torch.no_grad():
    probs = torch.sigmoid(model(X))
    preds = (probs > 0.5).float()
print("XNOR probs:\n", probs.round(decimals=3))
print("XNOR preds:\n", preds)
