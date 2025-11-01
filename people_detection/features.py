import numpy as np
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
import cv2

def extract_features(image):
    """Extract HOG + LBP features from an RGB image."""
    gray = rgb2gray(cv2.resize(image, (128, 128)))
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8), 
                          cells_per_block=(2, 2), visualize=True)
    lbp_features = local_binary_pattern(gray, 48, 6, method='uniform').flatten()
    return np.concatenate((hog_features, lbp_features)).reshape(1, -1)