import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

print("Is CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("Current GPU: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_transforms = transforms.Compose([transforms.ToTensor()])
train_data = torchvision.datasets.FashionMNIST('fashion_data', train=True, download=True, transform=all_transforms)
test_data  = torchvision.datasets.FashionMNIST('fashion_data', train=False, transform=all_transforms)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

def mnist_label(label):
    output_mapping = {
        0: "T-shirt/Top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot"
        }
    label = label.item() if type(label) == torch.Tensor else label
    return output_mapping[label]

class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        self.convlayer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fully_connected_layer = nn.Sequential(
        nn.Linear(in_features=64*14*14, out_features=2048),
        nn.Dropout2d(0.20),
        nn.ReLU(),
        nn.Linear(in_features=2048, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=32),
        nn.ReLU(),
        nn.Linear(in_features=32, out_features=10),
        )
    
    def forward(self,x):
        out = self.convlayer(x)
        out = out.view(out.size(0), -1)
        out = self.fully_connected_layer(out)
        return out

model = FashionCNN().to(device)

error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
lstlosses = []
lstiterations = []
lstaccuracy = []

predictions_list = []
labels_list = []
num_epochs = 1
num_batches = 0
batch_size = 100

for epoch in range(num_epochs):
    print(f'Epoch: {epoch+1} of {num_epochs}')
    for images, labels in train_loader:
        train = Variable(images).to(device)
        labels = Variable(labels).to(device)
        outputs = model(train)
        loss = error(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_batches += 1

        if num_batches % batch_size == 0:
            total = 0
            matches = 0

            for images, labels in test_loader:
                labels = labels.to(device)
                labels_list.append(labels)
                test = Variable(images).to(device)
                output = model(test)

                predictions = torch.max(outputs, 1)[1]
                predictions_list.append(predictions)

                matches += (predictions == labels).sum()
                total += len(labels)
            print(labels)
            print(predictions)
            accuracy = matches * 100 / total
            lstlosses.append(loss.data)
            lstiterations.append(num_batches)
            lstaccuracy.append(accuracy)

        if not (num_batches % batch_size):
            print(f'Iteration: {num_batches}, Loss: {loss.data}, Accuracy:{accuracy:.2f}%')