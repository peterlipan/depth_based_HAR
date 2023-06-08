import numpy as np
from skimage.feature import hog


def cal_hog_features(image, num_orientations, img_size):
    """
    Calculate the histogram of the Orientated Gradients of the input image.
    """
    # set pixels_per_cell = img_size to reduce the complexity
    hog_features = hog(image, orientations=num_orientations, pixels_per_cell=(img_size, img_size),
                       cells_per_block=(1, 1), visualize=False, feature_vector=False, channel_axis=-1)

    hog_features = hog_features.reshape(-1, num_orientations).sum(axis=0)

    return hog_features
