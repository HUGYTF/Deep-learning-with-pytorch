import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

print("Is CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("Current GPU: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(3, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn. BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)
       
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        output = self.model(x)
        return output.view(-1, 1)
    
class CoupleDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = []

        for file_name in os.listdir(folder_path):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_files.append(os.path.join(folder_path, file_name))
                    
    def __len__(self):
        return len(self.image_files) 
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        return image

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    plt.imshow(image)
    plt.show()

# Load the custom dataset.
folder_path1 = 'D:\\Python_Project\\Deep_Learning_Torch\\training_data'

transform = transforms.Compose([transforms.Resize((512,512)), 
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(10),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.Lambda(lambda x: x/255.0)
                                ])

train_dataset = CoupleDataset(folder_path1, transform=transform)
train_data = DataLoader(train_dataset, batch_size=16, shuffle=True)

netG = Generator().to(device)
netD = Discriminator().to(device)

criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=0.001)
optimizerD = optim.Adam(netD.parameters(), lr=0.001)

num_epochs = 1000

for epoch in range(num_epochs):
    for images in train_data:
        real_images = images.to(device)
        real_labels = torch.ones(images.size(0), 1).to(device)
        fake_labels = torch.zeros(images.size(0), 1).to(device)

        outputs = netD(real_images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z =torch.randn(images.size(0), 3, 1, 1).to(device)
        fake_images = netG(z)
        outputs = netD(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        
        d_loss = d_loss_real + d_loss_fake
        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        outputs = netD(fake_images)
        g_loss = criterion(outputs, real_labels)

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

    
        print(f'Epoch[{epoch + 1}]:'
                f'D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f},'
                f'D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')
    
im_convert(fake_images[0])