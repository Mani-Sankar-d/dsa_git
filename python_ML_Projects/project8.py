import numpy as np
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', as_frame=False)
X_mnist, y_mnist = mnist.data, mnist.target
X_train, y_train = X_mnist[:50_000],y_mnist[:50_000]
X_valid, y_valid = X_mnist[50_000:60_000],y_mnist[50_000:60_000]
X_test, y_test = X_mnist[60_000:],y_mnist[60_000:]

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
svm_clf = LinearSVC(max_iter=100, tol=20, dual=True, random_state=42)
mlp_clf = MLPClassifier(random_state=42)
estimators = [random_forest_clf,extra_trees_clf,svm_clf,mlp_clf]
for e in estimators:
    e.fit(X_train,y_train)
named_estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
    ("mlp_clf", mlp_clf),
]
blender = RandomForestClassifier(n_estimators=100, random_state=42)
X_valid_predictions = np.empty((len(X_valid),len(estimators)),dtype=object)
for index, estimator in enumerate(estimators):
    X_valid_predictions[:,index] = estimator.predict(X_valid)

blender.fit(X_valid_predictions, y_valid)
from sklearn.ensemble import StackingClassifier
stack_clf =StackingClassifier(named_estimators, final_estimator=blender)
stack_clf.fit(X_mnist[:60000],y_mnist[:60000])
print(stack_clf.score(X_test, y_test))