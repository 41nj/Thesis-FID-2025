import argparse
import random
from torchvision import models, transforms
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader, Subset,SubsetRandomSampler
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import numpy as np
import custom_fid
import glob
import json
from datetime import datetime
import sys
from get_images_for_specific_label import process_class_images
import clip
from clip.model import CLIP
from einops import rearrange

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
parser.add_argument('-HM', type=bool, default=False, help='')
parser.add_argument('-imgnet', type=bool, default=False, help='')
parser.add_argument('--class1', type=str, help='')
parser.add_argument('--class2', type=str, help='')
parser.add_argument('--chooseLayer', type=bool, default=False, help='')





args = parser.parse_args()

# Assign arguments to variables
dirname1, dirname2, num, seed, method, model_name, heatmaps_img, imgnet, class1, class2, cl = args.dirname1, args.dirname2, args.N, args.seed, args.method, args.model, args.HM, args.imgnet, args.class1, args.class2, args.chooseLayer

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(seed)

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
    Get random image paths from the directory
    """
    image_extensions = ['**/*.JPEG', '**/*.jpg', '**/*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dir, ext), recursive=True))
    if len(image_paths) == 0:
        print(f"Warning: No images found in {dir}")
        return []
    random.shuffle(image_paths)  # Shuffle the paths randomly
    return image_paths[:num]

def load_model(model_name):
    """
    Load the model for FID calculation
    
    :param model: Name of the model to load
    :return: Model with fc layer removed
    """
    choosen_model = None
    if model_name == 'inception_v3':
        choosen_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        
        choosen_model.fc = nn.Identity()
        
        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif model_name == 'resnet50':
        choosen_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        choosen_model.fc = nn.Identity()
       
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    elif model_name == 'clip':
        choosen_model, preprocess = clip.load("ViT-B/32", device="cpu")
            
    else:
        raise ValueError('Unknown model: {}'.format(model_name))
    
    choosen_model.eval()
    return choosen_model, preprocess

def get_clip_image_features(image_paths, preprocess, model, batch_size):
    #normal  
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [preprocess(Image.open(path)).unsqueeze(0) for path in batch_paths]
        images = torch.cat(images).to(device)
        
        with torch.no_grad():
            features = model.encode_image(images)  
            all_features.append(features.cpu().numpy())  
            
    return np.concatenate(all_features, axis=0)

#transformation
"""
    all_features = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = [preprocess(Image.open(path)).unsqueeze(0) for path in batch_paths]
        images = torch.cat(images).to(device)
        
        noisy_images = []
        for image in images:
            # Bild in numpy Array umwandeln
            image_array = np.array(image.permute(1, 2, 0).cpu())  # HWC für numpy (Height, Width, Channels)
            
            # Rauschen hinzufügen
            noisy_image_array = np.clip(image_array + 100, 0, 255)
            
            # Umwandeln zurück in Tensor
            noisy_image_tensor = torch.tensor(noisy_image_array).permute(2, 0, 1).float()  # CHW für Tensor (Channels, Height, Width)
            noisy_images.append(noisy_image_tensor)

        # Zu einem Batch zusammenfassen
        noisy_images = torch.stack(noisy_images).to(device)
        
        with torch.no_grad():
            # Features für Originalbilder
            original_features = model.encode_image(images)  
            # Features für verrauschte Bilder
            noisy_features = model.encode_image(noisy_images)
            
            # Speichern
            all_features.append(original_features.cpu().numpy())
            all_features.append(noisy_features.cpu().numpy())

    return np.concatenate(all_features, axis=0)"""

class CustomInceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        # extract at....
        self.features = nn.Sequential(
            self.inception.Conv2d_1a_3x3,
            self.inception.Conv2d_2a_3x3,
            self.inception.Conv2d_2b_3x3,
            self.inception.maxpool1,
            self.inception.Conv2d_3b_1x1,
            self.inception.Conv2d_4a_3x3,
            self.inception.maxpool2
            #self.inception.Mixed_5b,
            #self.inception.Mixed_5c,
            #self.inception.Mixed_5d,
            #self.inception.Mixed_6a,
            #self.inception.Mixed_6b,
            #self.inception.Mixed_6c,
            #self.inception.Mixed_6d,
            #self.inception.Mixed_6e
        )

        #self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        #x = self.pool(x)
        x = rearrange(x, "b c h w -> b (c h w)")
        return x
    
def get_res_inc_image_features(model, loader):
    features = []
            
    for batch in loader:
        with torch.no_grad():
            

            # NORMAL
            feature = model(batch.to(device)).cpu().numpy()
            features.append(feature)
            
            #NOISE
            """
            noisy_images = []
            for image in batch:
                image_array = np.array(image.permute(1, 2, 0).cpu())  # HWC für numpy (Height, Width, Channels)
                
                noisy_image_array = np.clip(image_array + 30, 0, 255)

                
                noisy_image_tensor = torch.tensor(noisy_image_array).permute(2, 0, 1).float()  # CHW für Tensor (Channels, Height, Width)
                noisy_images.append(noisy_image_tensor)

            noisy_images = torch.stack(noisy_images).to(device)  # Zu einem Batch zusammenfassen
            noisy_features = model(noisy_images).cpu().numpy()
            real_features.append(noisy_features)"""
        del batch  
        torch.cuda.empty_cache()  
    #print(features.shape)
    features = np.concatenate(features, axis=0)
    print(features.shape)

    return features

def calculate_fid(model_name, dirname1, dirname2, num, method, batch_size):
    """
    Calculate FID score using the given method
    """
    model, preprocess = load_model(model_name)

    if imgnet:
    # Fetch all images for "class1" and "class2" 

        real_images, _ = process_class_images(
            num,
            xml_folder='val',
            image_folder=dirname1,
            category=class1,
            class_folder= class1
        )

        generated_images, _ = process_class_images(
            num,
            xml_folder='val',
            image_folder=dirname2,
            category=class2,
            class_folder =class2
        )

    else:
        real_images = get_all_paths(dirname1, num)
        generated_images = get_all_paths(dirname2, num)

    real_dataset = ImageDataset(real_images, preprocess)
    gen_dataset = ImageDataset(generated_images, preprocess)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if method == 'torchmetrics':
        
        if cl:
            model= CustomInceptionV3().eval()
        
        fid_metric = FrechetInceptionDistance(feature=model, normalize=True).to(device)
        for batch in real_loader:
            fid_metric.update(batch.to(device), real=True)
        for batch in gen_loader:
            fid_metric.update(batch.to(device), real=False)
        fid_score = fid_metric.compute().item()

    elif method == 'custom':
        if cl:
            model = CustomInceptionV3().eval()

        model.to(device)

        fid_custom = custom_fid.FIDCalculator(192) #manual

        if model_name == "clip":
            
            real_features = get_clip_image_features(real_images, preprocess, model, batch_size)
            gen_features = get_clip_image_features(generated_images, preprocess, model, batch_size)

            

        else:
            gen_features = get_res_inc_image_features(model,gen_loader)
            print(gen_features.shape)
            real_features = get_res_inc_image_features(model,real_loader)
            print(real_features.shape)
                  

        fid_custom.update(real_features, gen_features)
        fid_score = fid_custom.compute_fid()

    print(f"FID score: {fid_score}")
    return fid_score, gen_loader, real_loader


if __name__ == '__main__':
    print(seed)

    fid_score, gen_loader, real_loader = calculate_fid(model_name, dirname1, dirname2, num, method, batch_size=256)


    # Configure logging
    # Configure logging
    if imgnet:

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "arguments": {
                "dirpath1": dirname1,
                "dirpath2": dirname2,
                "num_of_samples": num,
                "seed": seed,
                "method": method,
                "model": model_name
            },
            "results": {
                "method": method,
                "class1": class1,
                "class2:": class2,
                "FID_Score": fid_score
            }
        }
    else:
        log_data = {
        "timestamp": datetime.now().isoformat(),
        "arguments": {
            "dirpath1": dirname1,
            "dirpath2": dirname2,
            "num_of_samples": num,
            "seed": seed,
            "method": method,
            "model": model_name
        },
        "results": {
            "method": method,
            "FID_Score": fid_score
        }
    }


    log_file = 'json_files/192_mix_logfile.json'

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



    if heatmaps_img:
        import create_heatmaps.create_heatmaps as HM

        acts_and_grads = HM.load_model_with_gradients()
        mean_reals, cov_reals = HM.compute_dataset_statistics(real_loader, acts_and_grads.network)
        mean_gen, cov_gen = HM.compute_dataset_statistics(gen_loader, acts_and_grads.network)
        #print(mean_gen, mean_reals, cov_gen, cov_reals)

        print(num)

        HM.generate_heatmaps_for_images(
            gen_loader=gen_loader,
            acts_and_grads=acts_and_grads,
            mean_reals=torch.tensor(mean_reals).to(device),
            cov_reals=torch.tensor(cov_reals).to(device),
            mean_gen=torch.tensor(mean_gen).to(device),
            cov_gen=torch.tensor(cov_gen).to(device),
            num_images=num,
            output_dir='./heatmaps2',
            num_HM = 35
        )
    