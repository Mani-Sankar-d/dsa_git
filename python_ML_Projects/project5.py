from sklearn.datasets import  load_wine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
wine = load_wine(as_frame=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(wine.data,wine.target, random_state=42)
param_grid= {
    'svc__gamma':loguniform(0.001,0.1),
    'svc__C': uniform(1,10)
}

model = make_pipeline( StandardScaler(), SVC())
rnd_search_cv = RandomizedSearchCV(model, param_grid, n_iter= 100, cv=5, random_state=42)


rnd_search_cv.fit(X_train, y_train)
print(rnd_search_cv.score(X_test,y_test))
