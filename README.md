# bachelor-thesis
This repository contains the code and analysis for my bachelor's thesis, which explores Inception models and the Fréchet Inception Distance (FID) metric. The work focuses on understanding the limitations of FID, investigating scenarios where the metric fails, and proposing improvements to enhance its reliability in evaluating generative models.
---

I don't yet know how correctly fid is implemented with torchmetrics:
The result was FID Score: 1.0496853590011597

The two attempts to implement FID myself were not so successful at the weekend, so mainly FID calculation, but also feature extraction. 

The first one was large and negative, probably because the matrix product of the covariances might have complex values. FID score: -142.8274688720703

With the second one, I wanted to cover the complex values and also try out a bit with the layers and approach feature extraction differently.
FID Score: 2685.00439453125

Unfortunately I had no more resources at google colab :(
