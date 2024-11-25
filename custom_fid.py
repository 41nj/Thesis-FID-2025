import numpy as np
import scipy.linalg

class FIDCalculator:
    """
    Class to compute the Frechet Inception Distance (FID) between two sets of features.
    """
    def __init__(self, feature_size=2048):
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
        :param real: Boolean indicating if the features belong to the real dataset (True) or generated dataset (False)
        """
        """# Update batch means and covariances
        batch_mu1 = np.mean(features1, axis=0, dtype=np.float64)
        batch_mu2 = np.mean(features2, axis=0, dtype=np.float64)
        batch_sigma1 = np.cov(features1, rowvar=False, bias=False, dtype=np.float64)
        batch_sigma2 = np.cov(features2, rowvar=False, bias=False, dtype=np.float64)

        # Determine whether we are updating for real or generated images
        if real:
            # Update statistics for real images
            self.mu1 = (self.mu1 * self.n_samples1 + batch_mu1 * features1.shape[0]) / (self.n_samples1 + features1.shape[0])
            self.sigma1 = (self.sigma1 * (self.n_samples1 - 1) + batch_sigma1 * (features1.shape[0] - 1) + 
                        (self.n_samples1 * features1.shape[0]) / (self.n_samples1 + features1.shape[0]) * 
                        (batch_mu1 - self.mu1).reshape(-1, 1) * (batch_mu1 - self.mu1)) / (self.n_samples1 + features1.shape[0])
            self.n_samples1 += features1.shape[0]
            #print(self.n_samples1)
        else:
            # Update statistics for generated images
            self.mu2 = (self.mu2 * self.n_samples2 + batch_mu2 * features2.shape[0]) / (self.n_samples2 + features2.shape[0])
            self.sigma2 = (self.sigma2 * (self.n_samples2 - 1) + batch_sigma2 * (features2.shape[0] - 1) + 
                        (self.n_samples2 * features2.shape[0]) / (self.n_samples2 + features2.shape[0]) * 
                        (batch_mu2 - self.mu2).reshape(-1, 1) * (batch_mu2 - self.mu2)) / (self.n_samples2 + features2.shape[0])
            self.n_samples2 += features2.shape[0]
            #print(self.n_samples2)"""
        self.mu1 = np.mean(features1, axis=0, dtype=np.float64)
        self.mu2 = np.mean(features2, axis=0, dtype=np.float64)
        self.sigma1 = np.cov(features1, rowvar=False, bias=False, dtype=np.float64)
        self.sigma2 = np.cov(features2, rowvar=False, bias=False, dtype=np.float64)

        self.n_samples1 = features1.shape[0]
        self.n_samples2 = features2.shape[0]



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
