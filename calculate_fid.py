import numpy as np
from scipy.linalg import sqrtm

class FIDCalculator:
    """
    Class to compute the Frechet Inception Distance (FID) between two sets of features.
    """
    def __init__(self, feature_size):
        """
        Constructor of the FIDCalculator class.

        :param feature_size: size of the feature vectors
        """
        self.feature_size = feature_size
        self.reset()
    
    def reset(self):
        """
        Reset the internal variables of the FIDCalculator.
        """
        self.n_samples1 = 0
        self.n_samples2 = 0
        self.mu1 = np.zeros(self.feature_size)
        self.mu2 = np.zeros(self.feature_size)
        self.sigma1 = np.zeros((self.feature_size, self.feature_size))
        self.sigma2 = np.zeros((self.feature_size, self.feature_size))

    def update(self, features1, features2):
        """
        Update the internal variables of the FIDCalculator with new features.
        
        :param features1: features of the first set of samples
        :param features2: features of the second set of samples
        """
        batch_size = features1.shape[0]
        self.mu1 = (self.n_samples1 * self.mu1 + features1.sum(axis=0)) / (self.n_samples1 + batch_size)
        self.mu2 = (self.n_samples2 * self.mu2 + features2.sum(axis=0)) / (self.n_samples2 + batch_size)
        self.sigma1 = (self.n_samples1 * self.sigma1 + np.dot(features1.T, features1)) / (self.n_samples1 + batch_size)
        self.sigma2 = (self.n_samples2 * self.sigma2 + np.dot(features2.T, features2)) / (self.n_samples2 + batch_size)
        
        self.n_samples1 += batch_size
        self.n_samples2 += batch_size

    def compute_fid(self):
        """
        Compute the Frechet Inception Distance (FID) between the two sets of features.
        """
        diff = self.mu1 - self.mu2
        covmean, _ = sqrtm(self.sigma1.dot(self.sigma2), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(self.sigma1 + self.sigma2 - 2 * covmean)
        return fid




    


