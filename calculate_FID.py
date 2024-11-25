import argparse
import random
from torchvision import models, transforms
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import numpy as np
import custom_fid
import glob
import create_heatmaps
import json
from datetime import datetime

# Argument parser
parser = argparse.ArgumentParser(
    prog='calculate_fid.py',
    description='Calculates FID score of two datasets'
)
parser.add_argument('dirname1', type=str, help='Path to directory with real images')
parser.add_argument('dirname2', type=str, help='Path to directory with generated images')
parser.add_argument('-N', type=int, default=10000, help='Number of samples to use for FID calculation')
parser.add_argument('--seed', type=int, default=random.randint(0, 10000), help='Random seed for sampling')
parser.add_argument('method', type=str, help='Method to calculate FID score: torchmetrics or custom')
parser.add_argument('--model', type=str, default='inception_v3', help='Choose model to calculate FID score')
args = parser.parse_args()

# Assign arguments to variables
dirname1, dirname2, num, seed, method, model = args.dirname1, args.dirname2, args.N, args.seed, args.method, args.model

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageDataset(Dataset):
    """
    PyTorch Dataset for loading images from a directory
    """
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            return img
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, 299, 299)  # Return a dummy image in case of error


def get_all_paths(dir, num):
    """
    Get all image paths in the directory
    """
    image_extensions = ['**/*.JPEG', '**/*.jpg', '**/*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dir, ext), recursive=True))
    if len(image_paths) == 0:
        print(f"Warning: No images found in {dir}")
    return image_paths[:num]


def load_model(model):
    """
    Load the model for FID calculation
    """
    choosen_model = None
    if model == 'inception_v3':
        choosen_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        choosen_model.fc = nn.Identity()
    else:
        raise ValueError('Unknown model: {}'.format(model))
    choosen_model.eval()
    
    return choosen_model


def calculate_fid(model, dirname1, dirname2, num, method, batch_size):
    """
    Calculate FID score using the given method
    """
    model = load_model(model)
    real_images = get_all_paths(dirname1, num)
    generated_images = get_all_paths(dirname2, num)

    random.seed(seed)
    random.shuffle(real_images)
    random.shuffle(generated_images)

    # Prepare DataLoader
    real_dataset = ImageDataset(real_images, transform)
    gen_dataset = ImageDataset(generated_images, transform)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if method == 'torchmetrics':
        fid_metric = FrechetInceptionDistance(feature=model, normalize=True).to(device)
        for batch in real_loader:
            fid_metric.update(batch.to(device), real=True)
        for batch in gen_loader:
            fid_metric.update(batch.to(device), real=False)
        fid_score = fid_metric.compute().item()

    elif method == 'custom':
        model.to(device)
        fid_custom = custom_fid.FIDCalculator(2048)
        real_features = []
        for batch in real_loader:
            with torch.no_grad():
                features = model(batch.to(device)).cpu().numpy()
                real_features.append(features)
            del batch  
            torch.cuda.empty_cache()  

        real_features = np.concatenate(real_features, axis=0)

        gen_features = []
        for batch in gen_loader:
            with torch.no_grad():
                features = model(batch.to(device)).cpu().numpy()
                gen_features.append(features)
            del batch  
            torch.cuda.empty_cache()  

        gen_features = np.concatenate(gen_features, axis=0)


        fid_custom.update(real_features, gen_features)
        fid_score = fid_custom.compute_fid()

    print(f"FID score: {fid_score}")
    return fid_score


if __name__ == '__main__':
    
    fid_score, real_loader, gen_loader = calculate_fid(model, dirname1, dirname2, num, method, batch_size=256)

    # Configure logging
    # Configure logging
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "arguments": {
            "dirpath1": dirname1,
            "dirpath2": dirname2,
            "num_of_samples": num,
            "seed": seed,
            "method": method,
            "model": model
        },
        "results": {
            "method": method,
            "FID_Score": fid_score
        }
    }

    log_file = 'FID_logfile.json'

    
    if os.path.exists(log_file):
        with open(log_file, 'r') as json_file:
            try:
                logs = json.load(json_file)  
            except json.JSONDecodeError:
                logs = []  
    else:
        logs = []

    logs.append(log_data)

    with open(log_file, 'w') as json_file:
        json.dump(logs, json_file, indent=4)

    acts_and_grads = create_heatmaps.load_model_with_gradients()
    mean_reals, cov_reals = create_heatmaps.compute_dataset_statistics(real_loader, acts_and_grads.network)
    mean_gen, cov_gen = create_heatmaps.compute_dataset_statistics(gen_loader, acts_and_grads.network)

    create_heatmaps.generate_heatmaps_for_images(
        gen_loader=gen_loader,
        acts_and_grads=acts_and_grads,
        mean_reals=torch.tensor(mean_reals),
        cov_reals=torch.tensor(cov_reals),
        mean_gen=torch.tensor(mean_gen),
        cov_gen=torch.tensor(cov_gen),
        num_images=args.N,
        output_dir='./heatmaps'
    )
