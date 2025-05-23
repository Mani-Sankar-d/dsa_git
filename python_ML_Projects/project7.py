import numpy as np
from sklearn.datasets import make_moons
X_moon, y_moon = make_moons(n_samples=10000, noise=0.4, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_moon, y_moon, test_size=0.2, random_state=42)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
params = {
    'max_leaf_nodes': list(range(2, 100)),
    'max_depth': list(range(1, 7)),
    'min_samples_split': [2, 3, 4]
}
grid_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=params, cv=3)
grid_cv.fit(X_train, y_train)
best_model= grid_cv.best_estimator_

from sklearn.model_selection import ShuffleSplit
n_trees = 1000
n_instances = 100
mini_sets = []
rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train)-n_instances, random_state=42)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train,y_mini_train))

from sklearn.base import clone

forest = [clone(best_model) for _ in range(n_trees)]
Y_pred = np.empty([n_trees,len(X_test)], dtype=np.uint8)

for tree, (X_mini_train,y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train,y_mini_train)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

from scipy.stats import mode
from sklearn.metrics import accuracy_score
y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)
print(accuracy_score(y_test,y_pred_majority_votes.reshape([-1])))