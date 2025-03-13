import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def mean_pooling(features, target_dim):
        factor = features.shape[1] // target_dim  
        return features.reshape(features.shape[0], target_dim, factor).mean(axis=2)

class FIDCalculator:
    """
    Class to compute the Frechet Inception Distance (FID) between two sets of features.
    """
    def __init__(self, feature_size):
        self.feature_size = feature_size
        self.reset()
    
    def reset(self):
        self.n_samples1 = 0
        self.n_samples2 = 0
        self.mu1 = np.zeros(self.feature_size)
        self.mu2 = np.zeros(self.feature_size)
        self.sigma1 = np.zeros((self.feature_size, self.feature_size))
        self.sigma2 = np.zeros((self.feature_size, self.feature_size))

    



    def update(self, features1, features2):
        """
        Update the statistics with the features from a batch.
        
        :param features1: Features of the real or generated images from the first dataset
        :param features2: Features of the real or generated images from the second dataset
        """
        
        
        self.mu1 = np.mean(features1, axis=0, dtype=np.float64)
        self.mu2 = np.mean(features2, axis=0, dtype=np.float64)

        #random
        #indices = np.random.choice(features1.shape[1], 128, replace=False)
        #random_reduced_features1 = features1[:, indices]
        #random_reduced_features2 = features2[:, indices]
        #self.sigma1 = np.cov(random_reduced_features1, rowvar=False, bias=False, dtype=np.float64)
        #self.sigma2 = np.cov(random_reduced_features2, rowvar=False, bias=False, dtype=np.float64)


        #pca
        #pca = PCA(n_components=128)
        #reduced_features1 = pca.fit_transform(features1)
        #reduced_features2 = pca.fit_transform(features2)
        #self.sigma1 = np.cov(reduced_features1, rowvar=False, bias=False, dtype=np.float64)
        #self.sigma2 = np.cov(reduced_features2, rowvar=False, bias=False, dtype=np.float64)

        #pooled
        #pooled_features1 = mean_pooling(features1, 128)
        #pooled_features2 = mean_pooling(features2, 128)
        #self.sigma1 = np.cov(pooled_features1, rowvar=False, bias=False, dtype=np.float64)
        #self.sigma2 = np.cov(pooled_features2, rowvar=False, bias=False, dtype=np.float64)


        #original
        self.sigma1 = np.cov(features1, rowvar=False, bias=False, dtype=np.float64)
        self.sigma2 = np.cov(features2, rowvar=False, bias=False, dtype=np.float64)


        

    def compute_fid(self):
        """
        Compute the Frechet Inception Distance (FID).
        """
        diff = self.mu1 - self.mu2
        epsilon = 1e-6  # to avoid numerical issues
        covmean, _ = scipy.linalg.sqrtm((self.sigma1 + epsilon * np.eye(self.feature_size)) @ 
                                         (self.sigma2 + epsilon * np.eye(self.feature_size)), disp=False)

        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff @ diff + np.trace(self.sigma1 + self.sigma2 - 2 * covmean)
        return fid