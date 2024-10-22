import os
import numpy as np
from PIL import Image, ImageFilter
import glob

def add_noise(images, noise_factor):
    """
    Add noise to a list of image arrays.
    
    :param images: list of image arrays (NumPy)
    :param noise_factor: strength of the added noise
    :return: list of noisy image arrays
    """
    print(noise_factor)
    noisy_images = []
    for img in images:
        noise = noise_factor * np.random.randn(*img.shape)  
        noisy_img = np.clip(img + noise, 0, 1)  # Add noise and clip to [0, 1]
        noisy_images.append(noisy_img)
    return noisy_images

def blur_images(image_paths, blur_radius):
    """
    Use a Gaussian blur filter on a list of images.
    
    :param image_paths: list of paths to image files
    :param blur_radius: radius for the Gaussian blur filter
    :return: list of blurred images
    """
    print(blur_radius)
    blurred_images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')  
        img_blurred = img.filter(ImageFilter.GaussianBlur(blur_radius))  
        blurred_images.append(img_blurred)
    return blurred_images

def load_images_from_folder(folder_path):

    image_paths = glob.glob(os.path.join(folder_path, "*.png"))
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img_array = np.asarray(img) / 255.0  # Normalize to [0, 1]
        images.append(img_array)
    return images, image_paths

def save_images(images, original_paths, save_folder):

    os.makedirs(save_folder, exist_ok=True)  
    for img_array, orig_path in zip(images, original_paths):
        img = Image.fromarray((img_array * 255).astype(np.uint8))  
        img_name = os.path.basename(orig_path)  
        img.save(os.path.join(save_folder, img_name))  

def process_images(folder_path, save_folder, noise_factor, blur_radius):
    """
    Process images by adding noise and blur.
    
    :param folder_path: path to the folder containing the images
    :param save_folder: path to the folder to save the processed images
    :param noise_factor: strength of the added noise
    :param blur_radius: radius for the Gaussian blur filter
    """
    print(noise_factor, " ", blur_radius)
    # load images
    images, image_paths = load_images_from_folder(folder_path)
    
    # add noise
    noisy_images = add_noise(images, noise_factor=noise_factor)
    
    # blur images
    blurred_images = blur_images(image_paths, blur_radius=blur_radius)
    
    # save images
    save_images(noisy_images, image_paths, os.path.join(save_folder, "noisy"))
    
    os.makedirs(os.path.join(save_folder, "blurred"), exist_ok=True)
    for img, img_path in zip(blurred_images, image_paths):
        img.save(os.path.join(save_folder, "blurred", os.path.basename(img_path)))  


if __name__ == "__main__":
    for digit in range(10):
        folder_path = f"mnist_digits/{digit}"  
        save_folder = f"processed_images/5_0.7/{digit}"  
        process_images(folder_path, save_folder, noise_factor=0.7, blur_radius=5)
   
