import os
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from PIL import Image

# Load MNIST
mnist = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

x_train = mnist.data.numpy()
y_train = mnist.targets.numpy()
x_test = mnist_test.data.numpy()
y_test = mnist_test.targets.numpy()

# Combine train and test datasets
x_data = np.concatenate((x_train, x_test))
y_data = np.concatenate((y_train, y_test))

# Create directories for each digit
for i in range(10):
    os.makedirs(f'mnist_digits/{i}', exist_ok=True)

# Save 50 images of each digit
for digit in range(10):
    count = 0
    for i, label in enumerate(y_data):
        if label == digit:
            img = Image.fromarray(x_data[i])
            img.save(f'mnist_digits/{digit}/{digit}_{count}.png')
            count += 1
            if count >= 50:
                break

