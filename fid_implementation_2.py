import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import numpy as np
import gc 
import torch.nn.functional as F 


# load pretrained Inception v3 model
model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
model.eval()

# function to choose layers for feature extraction
def get_inception_features(images, model):
    with torch.no_grad():
        x = model.Conv2d_1a_3x3(images)
        x = model.Conv2d_2a_3x3(x)
        x = model.Conv2d_2b_3x3(x)
        x = model.maxpool1(x)
        x = model.Conv2d_3b_1x1(x)
        x = model.Conv2d_4a_3x3(x)
        x = model.maxpool2(x)
        x = model.Mixed_5b(x)
        x = model.Mixed_5c(x)
        x = model.Mixed_5d(x)
        x = model.Mixed_6a(x)
        x = model.Mixed_6b(x)
        x = model.Mixed_6c(x)
        x = model.Mixed_6d(x)
        x = model.Mixed_6e(x)
        x = model.Mixed_7a(x)
        x = model.Mixed_7b(x)
        features = model.Mixed_7c(x)  # get features from this layer

        # average pooling and flatten
        features = F.adaptive_avg_pool2d(features, (1, 1))  # reduce to 1x1
        features = features.view(features.size(0), -1)  # flatten

        return features

# push model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# transformations for MNIST (resize to 299x299 and normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # MNIST is grayscale so we need to convert to 3 channels
    transforms.Resize(299),  
    transforms.ToTensor(),  
])

# load MNIST dataset (train & test to simulate real and generated images)
real_dataset = MNIST(root="data", train=True, download=True, transform=transform)
fake_dataset = MNIST(root="data", train=False, download=True, transform=transform)

# for batch access using DataLoader
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

def calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake):
    """calculate FID score based on features from real and fake images"""
    # difference of means
    diff = mu_real - mu_fake

    # calculate trace of covariance matrices and their product
    covmean = sqrtm(sigma_real @ sigma_fake)
    
    # Check if covmean is complex
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # calculate FID
    fid = diff @ diff + torch.trace(sigma_real + sigma_fake - 2 * torch.tensor(covmean))
    return fid.item()

# initialize variables for mean, covariance and number of samples
mu_real = 0
sigma_real = 0
num_samples_real = 0

mu_fake = 0
sigma_fake = 0
num_samples_fake = 0


# feature extraction, instead of using the FrechetInceptionDistance class

# batch processing for real images
for batch_idx, (images, _) in enumerate(real_loader):
    images = images.to(device)
    features = get_inception_features(images, model).detach().cpu()
    
    # reshape features to (batch_size, channels * height * width)
    features = features.reshape(features.size(0), -1) # flatten spatial dimensions

    batch_mean = torch.mean(features, dim=0)
    batch_size = features.shape[0]

    # update total mean incrementally
    mu_real = (num_samples_real * mu_real + batch_size * batch_mean) / (num_samples_real + batch_size)

    # update total covariance incrementally
    if num_samples_real > 0:
        delta = batch_mean - mu_real  
        sigma_real += (batch_size/(num_samples_real+batch_size)) * (torch.cov(features.T) + delta.reshape(-1,1) @ delta.reshape(1,-1) * (num_samples_real/batch_size))
    else:
        sigma_real = torch.cov(features.T)
    
    num_samples_real += batch_size
    del images, features  # delete to free memory
    gc.collect()  # garbage collection

# batch processing for fake images
for batch_idx, (images, _) in enumerate(fake_loader):
    images = images.to(device)
    features = get_inception_features(images, model).detach().cpu()

    # reshape features to (batch_size, channels * height * width)
    features = features.reshape(features.size(0), -1) # flatten spatial dimensions

    batch_mean = torch.mean(features, dim=0)
    batch_size = features.shape[0]

    mu_fake = (num_samples_fake * mu_fake + batch_size * batch_mean) / (num_samples_fake + batch_size)
    
    # update total covariance incrementally
    if num_samples_fake > 0:
        delta = batch_mean - mu_fake  
        sigma_fake += (batch_size/(num_samples_fake+batch_size)) * (torch.cov(features.T) + delta.reshape(-1,1) @ delta.reshape(1,-1) * (num_samples_fake/batch_size))
    else:
        sigma_fake = torch.cov(features.T)

    num_samples_fake += batch_size
    del images, features
    gc.collect()


# calculate FID score
fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
print(f"FID Score: {fid_score}")