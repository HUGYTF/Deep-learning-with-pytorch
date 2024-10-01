import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import seaborn as sns

data_train = pd.read_csv('possum.csv')
data_test = pd.read_csv('possum.csv')

x = data_train.iloc[ : , 12]
y = data_train.iloc[ : , 13]

sns.scatterplot(x=x, y=y, size=0.5)
plt.show()

# Simple sequential model (an input layer, activation function, output layer)
model = nn.Sequential(
    nn.Linear(1,1),
    nn.ReLU(),
    nn.Linear(1,1)
)

# Convert to tensor.
x = torch.from_numpy(x.values).float().unsqueeze(1)
y = torch.from_numpy(y.values).float().unsqueeze(1)

# scale the data in the 0–1 range.
x = (x-x.min())/(x.max()-x.min())
y = (y-y.min())/(y.max()-y.min())

#  Initialize mean squared error (MSE) loss function and an optimizer and
#  use stochastic gradient descent for updating the weights.
lossfunction = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# Use a for loop to iterate over 50 epochs, keeping track of losses.
loss_history = []

for epoch in range(50):
    pred = model(x)
    loss = lossfunction(pred, y)

    loss_history.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

sns.lineplot(loss_history, marker='o')
plt.show()


