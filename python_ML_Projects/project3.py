import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
train_data = pd.read_csv('./datasets/titanic/train.csv')
test_data = pd.read_csv('./datasets/titanic/test.csv')
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline([('ordinal_encoder', OrdinalEncoder()), ('imputer', SimpleImputer(strategy='most_frequent')),
                         ('cat_encoder', OneHotEncoder(sparse=False))])
num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]
preprocess_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs), ('cat', cat_pipeline, cat_attribs)])
X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data['Survived']

# forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

# forest_clf.fit(X_train, y_train)#81%
# y_pred = forest_clf.predict(X_test)


svc_clf = SVC(gamma='auto')
X_test = preprocess_pipeline.transform(test_data)
svm_scores = cross_val_score(svc_clf, X_train, y_train, cv=10)
print(svm_scores.mean())#82.5%

