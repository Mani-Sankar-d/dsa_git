import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV
from scipy.ndimage import shift
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', as_frame=False)


def plot_image(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='binary')
    plt.axis('off')


def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode='constant')
    return shifted_image.reshape([-1])



X, y = mnist.data, mnist.target
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

knn_clf = KNeighborsClassifier(n_neighbors=4, weights='distance') # from previous test

X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dy, dx))
        y_train_augmented.append(label)
X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]
knn_clf.fit(X_train_augmented, y_train_augmented)
augmented_accuracy = knn_clf.score(X_test, y_test)
print(augmented_accuracy)