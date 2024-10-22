import os
import random
from PIL import Image
from torchvision import transforms
import glob

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def random_crop_images(input_dir, output_dir, crop_ratios):
    """
    cuts images randomly with a given ratio and saves them in the output directory
    """
    image_paths = glob.glob(os.path.join(input_dir, "*.png"))

    for ratio in crop_ratios:
        crop_output_dir = os.path.join(output_dir, f"crop_{int(ratio * 100)}")
        ensure_dir(crop_output_dir)

        for img_path in image_paths:
            img = Image.open(img_path).convert('RGB')
            img_name = os.path.basename(img_path)
            
            # get image size and calculate new size
            width, height = img.size
            new_width, new_height = int(width * ratio), int(height * ratio)
            
            # random coordinates for cropping
            left = random.randint(0, width - new_width)
            top = random.randint(0, height - new_height)
            right = left + new_width
            bottom = top + new_height
            
            # cut image and save it
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(os.path.join(crop_output_dir, img_name))



def main():
    input_dir = 'mnist_images'  
    base_output_dir = 'crops'

    crop_output_dir = os.path.join(base_output_dir, 'cropped_images')
    random_crop_images(input_dir, crop_output_dir, crop_ratios=[0.2, 0.5, 0.7])


if __name__ == '__main__':
    main()
