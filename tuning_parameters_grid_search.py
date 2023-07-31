from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

x = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

TEST_SIZE = .2
RANDOM_STATE = 1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
param_range = [.0001, .001, .01, .1, 1, 10, 100, 1000]
param_grid = [
    # linear SVM parameters
    {
    'svc__C': param_range, 
    'svc__kernel': ['linear']
    },
    # RBF kernel SVM parameters
    {
    'svc__C': param_range, 
    'svc__gamma': param_range, 
    'svc__kernel': ['rbf']
    }]

gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=-1)
gs = gs.fit(x_train, y_train)
print(gs.best_score_)
print(gs.best_params_)
clf = gs.best_estimator_
clf.fit(x_train, y_train)
print('Test accuracy: %.3f' % clf.score(x_test, y_test))
