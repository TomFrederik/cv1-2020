import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from sklearn import svm
from classification import calc_features, create_dataset, train_svm

#################
# Parameters
#################
vocab_size = 400
# train / test samples used per class
n_train = 50
n_test = 800

#################
# Constants, initialization
#################

class_names = ['airplane', 'bird', 'ship', 'horse', 'car']
class_nbrs = [1,2,3,4,5]

dirs = {
    "train": "./data/train/",
    "test": "./data/test/"
}
class_dirs = {k: [v + str(nbr) + '/' for nbr in class_nbrs] for k, v in dirs.items()}

dictionary = np.load('./dictionaries/k_{}.npy'.format(vocab_size))

sift = cv.xfeatures2d.SIFT_create()


#################
# Evaluation
#################

def calculate_map(classifier, x, y):
    """Calculate the mean average precision over some test data.

    Args:
        classifier: trained sklearn.svm.LinearSVC instance
        x: features
        y: labels (-1 or 1 for not in class / in class)"""
    scores = classifier.decision_function(x)
    # sort scores in descending order
    indices = np.argsort(-scores)

    result = 0
    f_c = 0
    for i, label in enumerate(y[indices]):
        if label == 1:
            f_c += 1
            result += f_c / (i + 1)
    # f_c is now the total number m_c of images with class c
    return result / f_c


if __name__ == "__main__":
    for i in class_nbrs:
        print("=====================================")
        print("Training for class {} ({})".format(i, class_names[i - 1]))
        classifier = train_svm(i, n_train)
        print("Evaluating")
        test_x, test_y = create_dataset(i, "test", n_test)
        score = calculate_map(classifier, test_x, test_y)
        print("Class {} ({}), maP: {:.3f}".format(i, class_names[i - 1], score))
