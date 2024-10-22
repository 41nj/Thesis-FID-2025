import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from os import path
import glob
import plotting
import calculate_fid

def get_inception_features(img_paths, model, transform, device):
    features = []
    model.eval()  
    with torch.no_grad():  
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')  # convert to RGB because InceptionV3 expects 3 channels
            img = transform(img).unsqueeze(0).to(device)  # transform and add batch dimension
            feature = model(img)  
            features.append(feature.cpu().numpy().flatten())  # features are saved as 1D array
    return np.array(features)


def test_with_inceptionV3(digits_real, digits_fake):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize InceptionV3 model
    inception_v3 = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    inception_v3.fc = nn.Identity()  # only features, no classification
    inception_v3.to(device)

    # transform as required by InceptionV3
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # define directories of the real and generated images
    current_script_path = path.abspath(__file__)
    current_dir = path.dirname(current_script_path)
    
    target_file_real = digits_real
    target_file_fake = digits_fake 
    
    target_path_real = path.join(current_dir, target_file_real)
    target_path_fake = path.join(current_dir, target_file_fake)

    real_image_paths = glob.glob(path.join(target_path_real, "*.png"))
    generated_image_paths = glob.glob(path.join(target_path_fake, "*.png"))
  
    # extract features from real and generated images
    features_real = get_inception_features(real_image_paths, inception_v3, transform, device)
    features_generated = get_inception_features(generated_image_paths, inception_v3, transform, device)

    # initialize FID calculator
    feature_size = 2048
    fid_calculator = fid.FIDCalculator(feature_size)

    # update the FID calculator with the features
    batch_size = 10
    for i in range(0, len(features_real), batch_size):
        fid_calculator.update(features_real[i:i+batch_size], features_generated[i:i+batch_size])

    # calculate FID score
    fid_value = fid_calculator.compute_fid()
    print("FID:", fid_value)
    return fid_value

def test_1(same):
    feature_size = 2048
    fid_calculator = fid.FIDCalculator(feature_size)

    # simulate features of 50 random real and generated images
    if same:
        features_fake = np.random.randn(50, feature_size)
        features_real = features_fake
    else:
        features_real = np.random.randn(50, feature_size)
        features_fake = np.random.randn(50, feature_size)


    # update the FID calculator with the features
    for i in range(5):
        fid_calculator.update(features_real[i*10:(i+1)*10], features_fake[i*10:(i+1)*10])

    # calculate FID score  
    fid_value = fid_calculator.compute_fid()
    print("FID:", fid_value)
    return fid_value

def run_tests():
    fid_values_single = []
    fid_values_multiple = []
    
    fid_value_0 = test_with_inceptionV3(digits_real="mnist_images", digits_fake="mnist_images")
    fid_values_multiple.append(fid_value_0)

    fid_value_1 = test_with_inceptionV3(digits_real="mnist_images", digits_fake="crops/cropped_images/crop_70")
    fid_values_multiple.append(fid_value_1)

    fid_value_2 = test_with_inceptionV3(digits_real="mnist_images", digits_fake="crops/cropped_images/crop_50")
    fid_values_multiple.append(fid_value_2)

    fid_value_3 = test_with_inceptionV3(digits_real="mnist_images", digits_fake="crops/cropped_images/crop_20")
    fid_values_multiple.append(fid_value_3)
    """
    fid_value_4 = test_with_inceptionV3(digits_real="mnist_images", digits_fake="mnist_images")
    fid_values_multiple.append(fid_value_4)
    
    fid_value_5 = test_with_inceptionV3(digits_real="mnist_digits/9", digits_fake="mnist_digits/5")
    fid_values_multiple.append(fid_value_5)

    fid_value_6 = test_with_inceptionV3(digits_real="mnist_digits/9", digits_fake="mnist_digits/6")
    fid_values_multiple.append(fid_value_6)

    fid_value_7 = test_with_inceptionV3(digits_real="mnist_digits/9", digits_fake="mnist_digits/7")
    fid_values_multiple.append(fid_value_7)

    fid_value_8 = test_with_inceptionV3(digits_real="mnist_digits/9", digits_fake="mnist_digits/8")
    fid_values_multiple.append(fid_value_8)

    fid_value_9 = test_with_inceptionV3(digits_real="mnist_digits/9", digits_fake="mnist_digits/9")
    fid_values_multiple.append(fid_value_9)

    
    
    # Run the second test and store the FID value
    fid_value_20 = test_1(same = True)
    fid_values_single.append(fid_value_20)
    fid_value_21 = test_1(same = False)
    fid_values_single.append(fid_value_21)
    """
    print(fid_values_multiple)
    return fid_values_single, fid_values_multiple

if __name__ == "__main__":
    #run_tests()    
    plotting.plot_fid_scores()
