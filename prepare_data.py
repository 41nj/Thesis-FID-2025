import torch
import torchvision
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import random
from PIL import ImageFilter
import math
from torch.utils.data import Dataset
from PIL import Image


class AddGaussianNoise:
    """Applies Gaussian noise to the image."""
    def __init__(self, mean=0.0, std=0.1):  
        self.mean = mean
        self.std = std

    def __call__(self, image):
        # Convert image to numpy array
        image_array = np.array(image).astype(np.float32) / 255.0
        
        # Add Gaussian noise
        noise = np.random.normal(self.mean, self.std, image_array.shape)
        noisy_image_array = np.clip(image_array + noise, 0, 1) * 255
        
        # Convert back to PIL image
        noisy_image = Image.fromarray(noisy_image_array.astype('uint8'))
        return noisy_image


class AddBlur:
    """Applies Gaussian blur to the image."""
    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=1.0))  

class BlockMasking:
    """Applies block masking to the image."""
    def __call__(self, img):
        np_img = np.array(img)
        h, w, _ = np_img.shape
        block_height, block_width = 4, 4  # Größe der Blöcke
        num_blocks = 16  # Anzahl der Blöcke
        for _ in range(num_blocks):
            x1 = random.randint(0, w - block_width)
            y1 = random.randint(0, h - block_height)
            np_img[y1:y1 + block_height, x1:x1 + block_width] = 0  # Block maskieren (schwarz setzen)
        return Image.fromarray(np_img)


class ImageSwirlTransform:
    """Applies swirl transformation to the image."""
    def __init__(self, disturbance=10, max_radius=None):
        self.disturbance = disturbance
        self.max_radius = max_radius  

    def __call__(self, image):
        # Convert PIL image to numpy array
        image_np = np.array(image)
        
        # Get the image dimensions (32x32 in your case)
        height, width, _ = image_np.shape
        center_x = width / 2
        center_y = height / 2
        
        if self.max_radius is None:
            self.max_radius = min(center_x, center_y)  

        # Create a copy of the image to apply the swirl effect 
        swirled_image = np.copy(image_np)  
        
        for x in range(width):
            for y in range(height):
                dx = x - center_x
                dy = y - center_y
                r = math.sqrt(dx**2 + dy**2)

                # transformation only if within the max radius
                if r < self.max_radius:
                    theta = math.atan2(dy, dx)
                    
                    # Apply the swirl effect 
                    theta_swirl = theta + (self.disturbance * r * math.log(2)) / 50  
                    
                    # Convert back to Cartesian coordinates (xorg, yorg)
                    xorg = int(center_x + r * math.cos(theta_swirl))
                    yorg = int(center_y + r * math.sin(theta_swirl))
                    
                    # check if the new coordinates are within image bounds
                    if 0 <= xorg < width and 0 <= yorg < height:
                        swirled_image[y, x] = image_np[yorg, xorg]
        
        # swirled image back to PIL format
        swirled_image_pil = Image.fromarray(swirled_image)
        
        return swirled_image_pil

class TransformedSubset(Dataset):
    """Applies a transformation to a subset of the dataset."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
                  
        return image, label


# Normalize Transformation
normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

# transformations
transform_original = transforms.Compose([transforms.ToTensor(), normalize])
transform_noise = transforms.Compose([AddGaussianNoise(), transforms.ToTensor(), normalize])
transform_blur = transforms.Compose([AddBlur(), transforms.ToTensor(), normalize])
transform_mask = transforms.Compose([BlockMasking(), transforms.ToTensor(), normalize])
transform_swirl = transforms.Compose([ImageSwirlTransform(disturbance=4, max_radius=25), transforms.ToTensor(), normalize])


trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_original)

# filter Classes
filtered_classes = ['dog', 'bird', 'airplane', 'truck']
filtered_indices = [i for i, label in enumerate(trainset.targets) if trainset.classes[label] not in filtered_classes]
trainset_filtered = torch.utils.data.Subset(trainset, filtered_indices)

# Split Dataset
trainset1_size = len(trainset_filtered) // 2
trainset2_size = len(trainset_filtered) - trainset1_size
trainset1_original, trainset2_transformations = torch.utils.data.random_split(trainset_filtered, [trainset1_size, trainset2_size])

# Apply Transformations
trainset1_original = TransformedSubset(trainset1_original, transform_original)
subset2_size = len(trainset2_transformations) // 4
trainset2_noise, trainset2_blur, trainset2_mask, trainset2_swirl = torch.utils.data.random_split(
    trainset2_transformations, [subset2_size, subset2_size, subset2_size, len(trainset2_transformations) - 3 * subset2_size])

trainset2_noise = TransformedSubset(trainset2_noise, transform_noise)
trainset2_blur = TransformedSubset(trainset2_blur, transform_blur)
trainset2_mask = TransformedSubset(trainset2_mask, transform_mask)
trainset2_swirl = TransformedSubset(trainset2_swirl, transform_swirl)

# Combine Datasets
combined_dataset = torch.utils.data.ConcatDataset([trainset1_original, trainset2_noise, trainset2_blur, trainset2_mask, trainset2_swirl])
#trainloader = torch.utils.data.DataLoader(combined_dataset, batch_size=4, shuffle=True, num_workers=2)

for i in range(len(combined_dataset)):
    image, label = combined_dataset[i]
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected image to be a torch.Tensor, but got {type(image)}")

print("All images in combined_dataset are tensors.")

# image saving
"""def save_transformed_image(dataset, index, transform, img_folder, filename):
    image, label = dataset[index]
    if transform:
        image = transform(image)
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    os.makedirs(img_folder, exist_ok=True)
    image.save(os.path.join(img_folder, filename))
    print(f"Image saved: {os.path.join(img_folder, filename)}")


img_folder = 'check_img_folder'
save_transformed_image(trainset2_noise, 0, None, img_folder, 'example_image_noise.png')
save_transformed_image(trainset2_blur, 0, None, img_folder, 'example_image_blur.png')
save_transformed_image(trainset2_mask, 0, None, img_folder, 'example_image_mask.png')
save_transformed_image(trainset2_swirl, 0, None, img_folder, 'example_image_swirl.png')"""

# verify sizes
print(f"Original Subset Size: {len(trainset1_original)}")
print(f"Noise Subset Size: {len(trainset2_noise)}")
print(f"Blur Subset Size: {len(trainset2_blur)}")
print(f"Mask Subset Size: {len(trainset2_mask)}")
print(f"Swirl Subset Size: {len(trainset2_swirl)}")

#check for unique labels
all_labels = [label for _, label in combined_dataset]
unique_labels = set(all_labels)
print("Unique labels in the combined dataset:", unique_labels)