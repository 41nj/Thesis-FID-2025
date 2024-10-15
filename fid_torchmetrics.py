import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# load pretrained model
model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
model.eval()

# move to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# transformations for MNIST (scale to 299x299 & normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize(299), 
    transforms.ToTensor(),
])

# load MNIST-dataset (train und test, to simulate real and generated images)
real_dataset = MNIST(root="data", train=True, download=True, transform=transform)
fake_dataset = MNIST(root="data", train=False, download=True, transform=transform)

# DataLoader for Batch-access
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

# intitialize FID (feature=2048)
fid = FrechetInceptionDistance(feature=2048).to(device)

def update_fid(loader, fid, real=True):
    """calculate FID score based on features from real and fake images"""
    with torch.no_grad():
        for images, _ in loader:
            # move to GPU if possible
            images = images.to(device)

            # scale to [0, 255] & convert to uint8
            images = (images * 255).type(torch.uint8)

            # update fid with scaled images
            fid.update(images, real=real)

# update fid with real and generated images
update_fid(real_loader, fid, real=True)
update_fid(fake_loader, fid, real=False)

# calculate FID
fid_score = fid.compute()
print(f"FID Score: {fid_score}")
