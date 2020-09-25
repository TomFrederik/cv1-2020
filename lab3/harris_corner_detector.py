import itertools
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image


def derivative(array, direction, sigma=1, mode="reflect"):
    """Take the first derivative by convolving with the derivative of a Gaussian."""
    if direction == "x":
        direction = 1
    elif direction == "y":
        direction = 0
    return gaussian_filter1d(array, axis=direction, sigma=sigma, mode=mode, order=1)


def cornerness(image, sigma_smooth=1, sigma_derivative=1):
    """Calculate the "cornerness" array H for an image."""
    I_x = derivative(image, "x", sigma_derivative)
    I_y = derivative(image, "y", sigma_derivative)
    A = gaussian_filter(I_x**2, sigma_smooth)
    B = gaussian_filter(I_x * I_y, sigma_smooth)
    C = gaussian_filter(I_y**2, sigma_smooth)

    return (A * C - B**2) - 0.04 * (A + C)**2


def harris_corner_detection(image, threshold, window_size, sigma_smooth=1, sigma_derivative=1):
    """Find corners in a grayscale image.

    Args:
        image: [H, W] ndarray of grayscale intensity (datatype should be float!)
        threshold: minimum cornerness value required for corners
        window_size: size of the window to use for finding local maxima (odd integer)
        sigma_smooth: sigma used for the smoothing Gaussian
        sigma_derivative: sigma used for the derivative of a Gaussian filter

    Returns:
        A tuple (H, rows, cols) of a cornerness array H (same shape as image),
        and a list of row and and columns indices of corners
    """
    assert window_size % 2 == 1, "window_size must be odd"
    H = cornerness(image, sigma_smooth, sigma_derivative)
    h, w = H.shape
    n = (window_size - 1) // 2
    rows, cols = [], []
    for i, j in itertools.product(range(h), range(w)):
        if H[i, j] < threshold:
            continue
        # Check whether there are any larger values in a (2n + 1) x (2n + 1) window
        min_row = max(i - n, 0)
        max_row = min(i + n + 1, h)
        min_col = max(j - n, 0)
        max_col = min(j + n + 1, w)
        if H[i, j] < np.max(H[min_row:max_row, min_col:max_col]):
            continue
        rows.append(i)
        cols.append(j)

    return H, rows, cols


def plot_corners(image, name, threshold, window_size=5, sigma_smooth=1, sigma_derivative=1):
    """Plot partial derivatives and image with corners.
    Arguments are the same as for harris_corner_detection where applicable.
    name is the filename to use for saving the image."""
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    I_x = derivative(image, "x", sigma_derivative)
    I_y = derivative(image, "y", sigma_derivative)
    plt.imshow(I_x)
    plt.title("I_x")
    plt.subplot(1, 3, 2)
    plt.imshow(I_y)
    plt.title("I_y")
    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap="gray")
    H, rows, cols = harris_corner_detection(image, threshold, window_size, sigma_smooth, sigma_derivative)
    print("Number of corners:", len(rows))
    plt.scatter(cols, rows, s=16)
    plt.tight_layout()
    plt.savefig("harris/{}.pdf".format(name))
    plt.show()
    # Additional plot to show the cornerness H
    #plt.imshow(image, cmap="gray")
    #plt.imshow(H, alpha=0.5, norm=colors.LogNorm())
    #plt.colorbar()
    #plt.show()


if __name__ == "__main__":
    ######################
    # set parameters here:
    ######################
    # thresholds are different for each image
    thresholds = [5000, 50000]
    # size of the window to look for local maxima, must be odd
    window_size = 13
    # sigma for gaussian smoothing
    sigma_smooth = 3
    # sigma used for the derivatives (using derivatives of Gaussians)
    sigma_derivative = 1

    paths = [
        "person_toy/00000001.jpg",
        "pingpong/0000.jpeg"
    ]
    names = ["person_toy", "pingpong"]

    for threshold, path, name in zip(thresholds, paths, names):
        image = np.asarray(Image.open(path).convert("L")).astype(float)
        plot_corners(image, name, threshold, window_size, sigma_smooth, sigma_derivative)

    image = np.asarray(Image.open(paths[0]).convert("L").rotate(45)).astype(float)
    plot_corners(image, "person_toy_45", thresholds[0], window_size, sigma_smooth, sigma_derivative)
    image = np.asarray(Image.open(paths[0]).convert("L").rotate(90)).astype(float)
    plot_corners(image, "person_toy_90", thresholds[0], window_size, sigma_smooth, sigma_derivative)
