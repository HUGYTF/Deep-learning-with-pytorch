from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

print("Is CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("Current GPU: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

folder_path = 'D:\Python_Project\Deep_Learning_Torch\gender_data\dataset'

transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])

class GenderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = []
        self.labels = []

        for label in ['MEN', 'WOMAN']:
            class_folder = os.path.join(folder_path, label)
            for file_name in os.listdir(class_folder):
                if file_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_files.append(os.path.join(class_folder, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_files) 
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
            label = 0 if self.labels[idx] == 'MEN' else 1
        
        return image, label
    
dataset = GenderDataset(folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

class GenderCNN(nn.Module):
   def __init__(self):
       super(GenderCNN, self).__init__()
       self.convlayer = nn.Sequential(
           nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2),
           nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, padding=1),
           nn.ReLU(), 
           nn.MaxPool2d(kernel_size=2, stride=2),
           nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
       self. fully_connected_layer = nn.Sequential(
           nn.Flatten(),
           nn.Linear(in_features=256*15*15, out_features=4096),
           nn.ReLU(),
           nn.Linear(in_features=4096, out_features=1024),
           nn.ReLU(),
           nn.Linear(in_features=1024, out_features=512),
           nn.ReLU(),
           nn.Linear(in_features=512, out_features=64),
           nn.ReLU(),
           nn.Linear(in_features=64, out_features=16),
           nn.ReLU(),
           nn.Linear(in_features=16, out_features=2),
           nn.LogSoftmax(dim=1)
       )
    
   def forward(self, x):
       out = self.convlayer(x)
       out= self.fully_connected_layer(out)
       return out

model = GenderCNN().to(device)

learning_rate = 0.5
num_epochs = 10

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for images, labels in dataloader:
        outputs = model(images.to(device))
        loss_value = loss(outputs, labels.to(device))

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels.to(device)).sum().item()
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value.item():.4f}, Accuracy: {accuracy:.2f}')
   
       
    