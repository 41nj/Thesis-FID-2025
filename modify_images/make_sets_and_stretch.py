import os
from torchvision import datasets, transforms
from PIL import Image
import random
from shutil import copyfile
import glob

def download_and_save_images(dataset_name, output_dir, num_images=50):
    """
    Downloads a specified number of images from a given dataset, applies transformations, and saves them to the specified output directory. 
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # convert grayscale images to RGB
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=3)])
    
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    elif dataset_name == 'FashionMNIST':
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    
    for i, (image, label) in enumerate(dataset):
        if i >= num_images:
            break
        img = transform(image)  # to PIL image
        img.save(os.path.join(output_dir, f'{dataset_name}_{i}.png'))

    print(f'{num_images} {dataset_name} images saved to {output_dir}')


def create_combined_datasets(real_set_dir, fake_set_dir, output_dir, ratios=[0.2, 0.5, 0.7]):
    """
    Creates combined datasets with different ratios of real and fake images.
    """

    real_images = glob.glob(os.path.join(real_set_dir, "*.png"))
    fake_images = glob.glob(os.path.join(fake_set_dir, "*.png"))

    for ratio in ratios:
        mixed_dir = os.path.join(output_dir, f'mixed_{int(ratio * 100)}')
        os.makedirs(mixed_dir, exist_ok=True)
        
        num_real = int(len(real_images) * ratio)
        num_fake = len(real_images) - num_real
        
        selected_real_images = random.sample(real_images, num_real)
        selected_fake_images = random.sample(fake_images, num_fake)
        
        for img_path in selected_real_images + selected_fake_images:
            img_name = os.path.basename(img_path)
            copyfile(img_path, os.path.join(mixed_dir, img_name))
            

def stretch_images(image_paths, output_dir, stretch_factors=[(2, 1), (1, 2), (2, 2)]):
    """
    creates stretched versions of images with different stretch factors.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for factor in stretch_factors:
        factor_dir = os.path.join(output_dir, f'stretch_{factor[0]}x{factor[1]}')
        os.makedirs(factor_dir, exist_ok=True)
        
        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img_stretched = img.resize((int(img.width * factor[0]), int(img.height * factor[1])))
            img_name = os.path.basename(img_path)
            img_stretched.save(os.path.join(factor_dir, img_name))
        
def main():

    mnist_dir = 'mnist_images'
    fashion_mnist_dir = 'fashion_mnist_images'
    combined_output_dir = 'combined_datasets'
    stretched_output_dir = 'stretched_images'

    download_and_save_images('MNIST', mnist_dir, num_images=50)
    download_and_save_images('FashionMNIST', fashion_mnist_dir, num_images=50)

    create_combined_datasets(mnist_dir, fashion_mnist_dir, combined_output_dir, ratios=[0.2, 0.5, 0.7])

    mnist_image_paths = glob.glob(os.path.join(mnist_dir, '*.png'))
    stretch_images(mnist_image_paths, stretched_output_dir)

if __name__ == '__main__':
    main()
