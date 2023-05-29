import numpy as np
from skimage.feature import hog


def cal_hog_features(image, num_orientations):
    """
    Calculate the histogram of the Orientated Gradients of the input image.
    """
    hog_features = hog(image, orientations=num_orientations, pixels_per_cell=(8, 8),
                       cells_per_block=(1, 1), visualize=False, feature_vector=False, channel_axis=-1)

    hog_features = hog_features.reshape(-1, num_orientations).sum(axis=0)

    # in case of all-black image
    if np.sum(hog_features) == 0:
        return hog_features
    else:
        return hog_features / np.sum(hog_features)
