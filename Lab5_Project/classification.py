import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn import svm



class_names = ['airplane', 'bird', 'ship', 'horse', 'car']
class_nbrs = [1,2,3,4,5]

dirs = {
    "train": "./data/train/",
    "test": "./data/test/"
}
class_dirs = {k: [v + str(nbr) + '/' for nbr in class_nbrs] for k, v in dirs.items()}

dictionary = np.load('./dictionaries/k_{}.npy'.format(400))

sift = cv.xfeatures2d.SIFT_create()


def calc_features(filename):
    """Calculates the histogram features for the given file.
    Uses sift and dictionary objects from global scope!"""
    img = cv.imread(filename)
    _, des = sift.detectAndCompute(img, None)
    distance = np.sum((des[:,None, :] - dictionary[None,:, :]) ** 2, axis=-1)
    cluster_assignments = np.argmin(distance, axis=-1)
    return np.bincount(cluster_assignments.astype(int), minlength=400)


def create_dataset(class_nbr, mode="train", num_images=50):
    """Create features and labels for a given class."""
    features = []
    labels = []

    for j, folder in enumerate(class_dirs[mode]):
        print(".", end="", flush=True)
        # get the first num_images images of each class
        _, _, files = next(os.walk(folder))
        files = [folder + f for f in files][:num_images]

        for cur_file in files:
            features.append(calc_features(cur_file))
            if j + 1 == class_nbr:
                labels.append(1)
            else:
                labels.append(-1)
    features = np.stack(features, axis=0)
    labels = np.array(labels)
    print("")
    return features, labels


def train_svm(class_nbr, num_images=50):
    """Returns a trained sklearn.svm.LinearSVC instance."""

    features, labels = create_dataset(class_nbr, "train", num_images)

    classifier = svm.LinearSVC()
    classifier.fit(features, labels)

    return classifier


if __name__ == "__main__":
    for i in class_nbrs:
        print("=====================================")
        print("Training for class {} ({})".format(i, class_names[i - 1]))
        classifier = train_svm(i, 50)
        print("Evaluating")
        test_x, test_y = create_dataset(i, "test", 50)
        score = classifier.score(test_x, test_y)
        print("Class {} ({}), mean accuracy: {:.2f}".format(i, class_names[i - 1], score))
