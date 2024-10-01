import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

print("Is CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("Current GPU: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data = pd.read_csv('Microsoft_Stock.csv')
data = data[['Date', 'Volume']]

data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y %H:%M:%S')

def chunkify(data, seq_length):
    chunks = []
    targets = []

    for i in range(len(data) - seq_length - 1):
        chunks.append(data[i:(i + seq_length)])
        targets.append(data[i + seq_length])
        
    return np.array(chunks), np.array(targets)

sc = MinMaxScaler()
training_data = sc.fit_transform(data[['Volume']].values.copy())

seq_length = 10
x, y = chunkify(training_data, seq_length)

train_size = int(len(y) * 0.8)
test_size = len(y) - train_size

dataX = torch.Tensor(x).to(device)
dataY = torch.Tensor(y)

trainX = torch.Tensor(x[0:train_size]).to(device)
trainY = torch.Tensor(y[0:train_size]).to(device)

testX = torch.Tensor(x[train_size:len(x)])
testY = torch.Tensor(y[train_size:len(y)])

class StockPrediction(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers, seq_length=seq_length):
        super(StockPrediction, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        ula, (h_out, _) = self.lstm(x, (h0, c0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)
        return out
    
num_epochs = 10000
learning_rate = 0.01

model = StockPrediction(1, 1, 1, 1).to(device)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    
    outputs = model(trainX)
    loss = criterion(outputs, trainY)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
model.eval()
train_predict = model(dataX)

data_predict = train_predict.cpu().data.numpy()
data_actual = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
data_actual = sc.inverse_transform(data_actual)

fig = plt.gcf().set_size_inches(1000, 8)
plt.plot(data_actual)
plt.plot(data_predict)
plt.suptitle('')
plt.legend(['Acutal cases', 'Predict cases'], loc='upper left')
plt.show()