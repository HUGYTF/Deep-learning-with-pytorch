import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

def load_image(image_path, max_size=512, shape=None):
    image = Image.open(image_path).convert('RGB')
    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)
    image = image.clip(0, 1)
    return image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
content = load_image('D:\Python_Project\Deep_Learning_Torch\style_transfer_data\content\image3.jpg').to(device)
style = load_image('D:\Python_Project\Deep_Learning_Torch\style_transfer_data\style\Vincent Van Gogh - The Starry Night (1888).jpg', shape=content.shape[-2:]).to(device)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features
    
def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def get_style_loss(style_features, target_features):
    style_loss = 0
    for sf, tf in zip(style_features, target_features):
        _, d, h, w = sf.size()
        gram_s = gram_matrix(sf)
        gram_t = gram_matrix(tf)
        style_loss += torch.mean((gram_s - gram_t) ** 2) / (d * h * w)
    return style_loss

def get_content_loss(content_features, target_features):
    content_loss = 0
    for cf, tf in zip(content_features, target_features):
        content_loss += torch.mean((cf - tf) ** 2)
    return content_loss


model = VGG().to(device).eval()

target = content.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target], lr=0.003)
style_weight = 1e6
content_weight = 1

for step in range(8000):
    target_features = model(target)
    content_features = model(content)
    style_features = model(style)

    content_loss = get_content_loss(content_features, target_features)
    style_loss = get_style_loss(style_features, target_features)
    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 500 == 0:
        print(f"Step {step}, Total loss: {total_loss.item()}")

final_img = im_convert(target)
plt.imshow(final_img)
plt.show()
