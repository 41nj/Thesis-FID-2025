from create_heatmaps.heatmap_utils import ActivationsAndGradients, compute_sensitivity_heatmap
from torchvision import models
import torch.nn as nn
import os
import torch
import numpy as np
import torch.nn.functional as F


imagenet_labels = {idx: entry.strip() for (idx, entry) in enumerate(open("imagenet_classes.txt"))}

def load_model_with_gradients():
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()  
    model.eval()
    return ActivationsAndGradients(model, {}, target_layer_name='Mixed_7c')

def validate_heatmap_focus(gen_image, acts_and_grads):
    model = acts_and_grads.network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    gen_image = gen_image.to(device)

    # calc prediction
    with torch.no_grad():
        logits = model(gen_image)  
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    # Top-1 class
    top1_class = int(np.argmax(probs))  # index 
    
    # check, if top1_class within 0-999 
    if top1_class < 0 or top1_class >= 1000:
        print(f"Warning: Class index {top1_class} is out of bounds, setting to 0.")
        top1_class = 0  # default
    
    top1_label = imagenet_labels.get(top1_class, "Unknown")
    top1_prob = probs[top1_class]

    # debugging
    print(f"Predicted class: {top1_label} (Probability: {top1_prob:.4f})")
    
    return top1_class, top1_label, top1_prob



def generate_heatmaps_for_images(gen_loader, acts_and_grads, mean_reals, cov_reals, mean_gen, cov_gen, num_images, output_dir, num_HM):
    os.makedirs(output_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k=0
    for i, batch in enumerate(gen_loader):
        
        gen_image = batch.to(device)
        for j in range(gen_image.size(0)):
            if k < num_HM:
            
                # create heatmap
                heatmap = compute_sensitivity_heatmap(
                    gen_image=gen_image[j:j+1],
                    acts_and_gradients=acts_and_grads,
                    mean_reals=mean_reals,
                    cov_reals=cov_reals,
                    mean_gen=mean_gen,
                    cov_gen=cov_gen,
                    num_images=num_images
                )

                # validate class
                top1_class, top1_label, top1_prob = validate_heatmap_focus(gen_image[j:j+1], acts_and_grads)

                # save heatmap
                heatmap.save(os.path.join(output_dir, f"heatmap_{i}_{j}_class_{top1_class}_{top1_label}.png"))
            k+=1

                

def compute_dataset_statistics(loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    features = []
    for batch in loader:
        with torch.no_grad():
            features.append(model(batch.to(device)).cpu().numpy())
    features = np.concatenate(features, axis=0)
    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)
    return mean, cov


