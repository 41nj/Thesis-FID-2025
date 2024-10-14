import torch
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Lade vortrainiertes Inception v3-Modell
model = torchvision.models.inception_v3(weights=torchvision.models.Inception_V3_Weights.IMAGENET1K_V1)
model.eval()

# Verschiebe das Modell auf die GPU, falls verfügbar
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Transformationen für MNIST (auf 299x299 skalieren und normalisieren)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  
    transforms.Resize(299), 
    transforms.ToTensor(),
])

# Lade den MNIST-Datensatz (train und test, um echte und generierte Bilder zu simulieren)
real_dataset = MNIST(root="data", train=True, download=True, transform=transform)
fake_dataset = MNIST(root="data", train=False, download=True, transform=transform)

# DataLoader für den Batch-Zugriff
real_loader = DataLoader(real_dataset, batch_size=32, shuffle=False)
fake_loader = DataLoader(fake_dataset, batch_size=32, shuffle=False)

# FID initialisieren (feature=2048 wegen v3)
fid = FrechetInceptionDistance(feature=2048).to(device)

def update_fid(loader, fid, real=True):
    """Aktualisiere FID mit echten oder generierten Bildern"""
    with torch.no_grad():
        for images, _ in loader:
            # Bilder auf GPU verschieben, falls verfügbar
            images = images.to(device)

            # Bilder auf [0, 255] skalieren und in uint8 umwandeln
            images = (images * 255).type(torch.uint8)

            # FID mit den skalierten Bildern aktualisieren
            fid.update(images, real=real)

# FID mit echten und generierten Bildern aktualisieren
update_fid(real_loader, fid, real=True)
update_fid(fake_loader, fid, real=False)

# Berechnung FID
fid_score = fid.compute()
print(f"FID Score: {fid_score}")