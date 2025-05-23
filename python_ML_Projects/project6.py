from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from  sklearn.metrics import mean_squared_error
import numpy as np
housing = fetch_california_housing()

X=housing.data
y=housing.target
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
svm_reg = make_pipeline(StandardScaler(), SVR())
param_distrib = {
    "svr__gamma": loguniform(0.001, 0.1),
    "svr__C": uniform(1, 10)
}
rnd_search_cv = RandomizedSearchCV(svm_reg, param_distrib,n_iter=100, cv=3, random_state=42)
rnd_search_cv.fit(X_train[:2000],y_train[:2000])
best_model = rnd_search_cv.best_estimator_
y_pred = (best_model.predict(X_test))
mse = mean_squared_error(y_test, y_pred)
print(np.sqrt(mse))