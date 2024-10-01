import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(10, 256)
        self.hidden = nn.Linear(256, 128)
        self.output = nn.Linear(128, 4)

    def forward(self, x):
        x = self.input(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.15)
        x = self.hidden(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.15)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x
    
mymodel = Net()

weather_data = pd.read_csv('weather_classification_data.csv')

X = weather_data.drop(labels='Weather Type', axis=1)

le = LabelEncoder()
X['Cloud Cover'] = le.fit_transform(X['Cloud Cover'])
X['Season'] = le.fit_transform(X['Season'])
X['Location'] = le.fit_transform(X['Location'])

sd = StandardScaler()
X = sd.fit_transform(X)

y = le.fit_transform(weather_data['Weather Type'])

data = torch.tensor(X).float()
labels = torch.tensor(y).long()

learning_rate = 0.5
num_epochs = 1500

lossfunc = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mymodel.parameters(), lr=learning_rate)

losses = []
accuracy = []

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mymodel(data)
    loss = lossfunc(outputs, labels)
    loss.backward()
    optimizer.step()

    losses.append(loss.detach())

    matches = (torch.argmax(outputs, axis=1) == labels).float()
    accuracy.append(100 * torch.mean(matches))

losses = [f'{loss.item():.2f}' for loss in losses]
accuracy = [f'{accuracy.item():.3f}%' for accuracy in accuracy]

print(losses)
print(accuracy)

'''
torch.save(mymodel.state_dict(), 'model.pth')
torch.save(mymodel, 'my_model.pth')
#When loading the model, we need to redefine the structure of the model before loading.

new_model = Net()
new_model.load_state_dict(torch.load('model.pth'))
new_model.eval()

# Now you can use new_model to make predictions.
'''