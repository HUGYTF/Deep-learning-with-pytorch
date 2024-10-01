from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from torchvision import models
import matplotlib.pyplot as plt
import threading
import queue

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


# Train the CNN model.
model = models.vgg16(pretrained=True).to(device)

for param in model.features.parameters():
    param.requires_grad = False

model.classifier[6].out_features = 2

# Define a queue to pass data to the plotting thread
plot_queue = queue.Queue()

def hook_fn(module, input, output):
    output = output.detach().cpu().numpy()
    output_img = output[0, 0, :, :]
    output_img = (output_img - output_img.min()) / (output_img.max() - output_img.min())
    plot_queue.put(output_img)

hook_handle = model.features[13].register_forward_hook(hook_fn)

def plot_thread():
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    while True:
        output_img = plot_queue.get()
        if output_img is None:
            break
        ax.imshow(output_img, cmap='gray')
        plt.title('Output of Conv Layer')
        plt.draw()
        plt.pause(0.001)
        ax.clear()

# Start the plotting thread
thread = threading.Thread(target=plot_thread)
thread.start()

learning_rate = 0.5
num_epochs = 10

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.classifier.parameters(), lr=learning_rate, momentum=0.5)

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
        correct = (predicted == labels.to(device)).sum().item()
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value.item():.4f}, Accuracy: {100 * accuracy:.2f}%')

# Signal the plotting thread to exit
plot_queue.put(None)
thread.join()

hook_handle.remove()