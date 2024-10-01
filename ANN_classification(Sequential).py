import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import torch, torch.nn as nn

weather_data = pd.read_csv('weather_classification_data.csv')

le = LabelEncoder()
X = weather_data.drop(labels='Weather Type', axis=1)
X['Cloud Cover'] = le.fit_transform(X['Cloud Cover'])
X['Season'] = le.fit_transform(X['Season'])
X['Location'] = le.fit_transform(X['Location'])
sd = StandardScaler()
X = sd.fit_transform(X)
y = le.fit_transform(weather_data['Weather Type'])

data = torch.tensor(X).float()
labels = torch.tensor(y).long()
print(data.size())
print(labels.size())

model = nn.Sequential(
    nn.Linear(10, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 4)
)

crossentropyloss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=.01)

maxiter = 3000
losses = []
accuracy = []

for epoch in range(maxiter):
    preds = model(data)
    loss = crossentropyloss(preds, labels)
    losses.append(loss.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    matches = (torch.argmax(preds, axis=1) == labels).float()
    accuracyPct = torch.mean(matches)
    accuracy.append(accuracyPct)

losses = [loss for loss in losses]
accuracy = [accuracy for accuracy in accuracy]

print(losses)
print(accuracy)
    

