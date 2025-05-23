from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
from os import X_OK
from sklearn import svm
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVR, SVR
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.cluster import KMeans


class arrtodf(BaseEstimator,TransformerMixin):
  def fit(self,X,y=None):
    return self
  def transform(self,X,y=None):
    feature_names=self.preprocessor.get_feature_names_out()
    return pd.DataFrame(X,columns=feature_names)


class KMeansClusterWithMedianPrice(BaseEstimator,TransformerMixin):
    def __init__(self,n_clusters=5):
        self.n_clusters=n_clusters
        self.kmeans=KMeans(n_clusters=self.n_clusters,random_state=42)
    def fit(self,X):
        self.kmeans.fit(X[['latitude','longitude']])
        X['cluster']=self.kmeans.predict(X[['latitude','longitude']])
        self.median_prices=X.groupby('cluster')['median_house_value'].median()
        return self
    def transform(self,X):
        X_new=X.copy()
        X_new['cluster']=self.kmeans.predict(X[['latitude','longitude']])
        X_new['cluster_median_price']=X_new['cluster'].map(self.median_prices)
        return X_new[['cluster_median_price']]


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
    with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()
clusterer=KMeansClusterWithMedianPrice(5)

numerical_pipeline = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
categorical_pipeline = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),
  ('onehot',OneHotEncoder(handle_unknown='ignore'))])
num_features=housing.select_dtypes(include=['number']).columns.tolist()
num_features.remove('median_house_value')
preprocessor = ColumnTransformer([('num',numerical_pipeline,num_features),
  ('categorical',categorical_pipeline,['ocean_proximity'])])
housing['cluster_median_price']=clusterer.fit_transform(housing)
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.01, 0.1, 1],
    'svm__epsilon': [0.1, 1, 10]
}
num_features.append('cluster_median_price')
feature_selector= SelectFromModel(LinearSVR(random_state=42),threshold='mean')
model=Pipeline([('preprocessor',preprocessor),('feature_selector',feature_selector),('svm',SVR(kernel='linear'))])
grid_search= GridSearchCV(model,param_grid,cv=5,scoring='neg_mean_squared_error',n_jobs=-1)

X=housing.drop('median_house_value',axis=1)
y=housing['median_house_value']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
grid_search.fit(X_train,y_train)
joblib.dump(grid_search.best_estimator_, "best_model.pkl")
