from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import matplotlib.pyplot as plt

print("Is CUDA available: ", torch.cuda.is_available())
print("Number of GPUs: ", torch.cuda.device_count())
print("Current GPU: ", torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.cuda.empty_cache()

vgg_model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device)

for param in vgg_model.parameters():
    param.requires_grad = False

# Correct layer indices for VGG19
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
content_layers = ['conv4_2']

style_weights = {layer: 1.0 / len(style_layers) for layer in style_layers}

def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def load_content_features(folder_path, target_size=256):
    images = []
    content_features = {}

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = image.convert("RGB")
            
            in_transform = transforms.Compose([
                                transforms.Resize(target_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), 
                                                    (0.229, 0.224, 0.225))])
            
            image = in_transform(image)[:3,:,:].unsqueeze(0).to(device)
            images.append(image)

            content_features[filename] = get_features(image, vgg_model, content_layers)
            
    return images, content_features

def load_style_features(folder_path, target_size=256):
    images = []
    style_features = {}

    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", '.jpeg', '.png')):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image = image.convert("RGB")
            
            in_transform = transforms.Compose([
                                transforms.Resize(target_size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), 
                                                    (0.229, 0.224, 0.225))])
            
            image = in_transform(image)[:3,:,:].unsqueeze(0).to(device)
            images.append(image)

            style_features[filename] = get_features(image, vgg_model, style_layers)
            
    return images, style_features

content_images, content_features = load_content_features('D:\Python_Project\Deep_Learning_Torch\anime_style_transfer_data\test\images')
style_images, style_features = load_style_features('D:\Python_Project\Deep_Learning_Torch\anime_style_transfer_data\test\labels')

generated_images = content_images[0].clone().requires_grad_(True).to(device)

def content_loss(gen_features, content_features):
    content_loss = nn.MSELoss()(gen_features['conv4_2'], content_features['conv4_2'])
    return content_loss

def gram_matrix(y):
    (b, c, h, w) = y.size()
    y = y.view(b, c, h * w)
    gram = torch.bmm(y, y.transpose(1, 2))
    gram =  gram.div_(c * h * w)
    return gram

def style_loss(gen_features, style_features, style_weights):
    style_loss = 0
    for layer in style_weights:
        gen_gram = gram_matrix(gen_features[layer])
        style_gram = style_features[layer]
        
        layer_style_loss = nn.MSELoss()(gen_gram, style_gram)
        
        style_loss += style_weights[layer] * layer_style_loss
    
    return style_loss

optimizer = optim.Adam([generated_images], lr=0.01)
num_epochs = 100
content_weight = 1e4
style_weight = 1e6

def closure():
    optimizer.zero_grad()   # Zero the gradients
    generated_features = get_features(generated_images, vgg_model, content_layers + style_layers)  # Forward pass
    content_loss_value = content_loss(generated_features, content_features)  # Compute content loss
    style_loss_value = style_loss(generated_features, style_features, style_weights)  # Compute style loss
    total_loss = content_weight * content_loss_value + style_weight * style_loss_value  # Combine losses
    total_loss.backward()  # Backward pass
    print(f"Total Loss: {total_loss.item()}")  # Optional: print the loss
    return total_loss  # Return the loss

for step in range(num_epochs):
    # Perform a single optimization step
    optimizer.step(closure)

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return image

plt.imshow(im_convert(generated_images))
plt.axis('off')
plt.show()

