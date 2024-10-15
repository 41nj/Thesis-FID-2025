import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import numpy as np

# load pretrained Inception v3 model
model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
model.eval()

# push model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# transformations for MNIST (resize to 299x299 and normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize(299),  
    transforms.ToTensor(),
])

# load MNIST dataset (train & test to simulate real and generated images)
real_dataset = MNIST(root="data", train=True, download=True, transform=transform)
fake_dataset = MNIST(root="data", train=False, download=True, transform=transform)

# for batch access use DataLoader
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)


def calculate_fid(real_features, fake_features):
    """calculate FID score based on features from real and fake images"""
    # Berechne Mittelwerte und Kovarianzmatrizen
    mu_real = torch.mean(real_features, dim=0)
    mu_fake = torch.mean(fake_features, dim=0)
    sigma_real = torch.cov(real_features.T)
    sigma_fake = torch.cov(fake_features.T)

    # difference of means
    diff = mu_real - mu_fake

    # calculate trace of covariance matrices and their product
    covmean = sqrtm(sigma_real @ sigma_fake)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # calculate FID
    fid = diff @ diff + torch.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid.item()

# extract features from real and fake images, instead of using the FrechetInceptionDistance class
real_features = []
fake_features = []

with torch.no_grad():
    for images, _ in real_loader:
        images = images.to(device)
        images = (images * 255).type(torch.uint8)
        features = model(images).detach().cpu()
        real_features.append(features)

    for images, _ in fake_loader:
        images = images.to(device)
        images = (images * 255).type(torch.uint8)
        features = model(images).detach().cpu()
        fake_features.append(features)

# concatenate features
real_features = torch.cat(real_features, dim=0)
fake_features = torch.cat(fake_features, dim=0)

#calculate FID score
fid_score = calculate_fid(real_features, fake_features)
print(f"FID Score: {fid_score}")


