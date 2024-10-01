from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast

# Check if GPU is available.
print("Is CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("Current GPU: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the custom dataset class for gender classification.
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

# Load the custom dataset.
folder_path = 'D:\Python_Project\Deep_Learning_Torch\gender_data\dataset'    
transform = transforms.Compose([transforms.Resize((512,512)), 
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.Lambda(lambda x: x/255.0)
                                ])

dataset = GenderDataset(folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the CNN model.
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
           nn.MaxPool2d(kernel_size=4, stride=4),
           nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, padding=1),
           nn.Sigmoid(),
           nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
       self. fully_connected_layer = nn.Sequential(
           nn.Flatten(),
           nn.Linear(in_features=256*15*15, out_features=4096),
           nn.ReLU(),
           nn.Linear(in_features=4096, out_features=128),
           nn.ReLU(),
           nn.Linear(in_features=128, out_features=16),
           nn.ReLU(),
           nn.Linear(in_features=16, out_features=2),
           nn.Softmax(dim=1)
       )
    
   def forward(self, x):
       out = self.convlayer(x)
       out= self.fully_connected_layer(out)
       return out

# Train the CNN model.
model = GenderCNN().to(device)

learning_rate = 0.5
num_epochs = 10

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scaler = GradScaler() 

for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images.to(device))
            loss_value = loss(outputs, labels.to(device))
        
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()

        _, predicted = torch.max(outputs, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value.item():.4f}, Accuracy: {100 * accuracy:.2f}%')