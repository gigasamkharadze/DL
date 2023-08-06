import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = [
    'Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity',
    'Hue', 'OD280/OD315 of diluted wines', 'Proline']

# drop 1 class
df_wine = df_wine[df_wine['Class label'] != 1]
y = df_wine['Class label'].values
x = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values

# encode class labels into binary format
le = LabelEncoder()
y = le.fit_transform(y)

# split data into 80% training and 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1, stratify=y)

# a decision tree classifier
tree = DecisionTreeClassifier(criterion='entropy', random_state=1, max_depth=None)

# create an ensemble of 500 decision trees fitted on different bootstrap samples of the training dataset
bag = BaggingClassifier(
    estimator=tree, n_estimators=500, max_samples=1.0, max_features=1.0, 
    bootstrap=True, bootstrap_features=False, n_jobs=1, random_state=1)

# train and evaluate the bagging classifier
bag = bag.fit(x_train, y_train)
y_train_pred = bag.predict(x_train)
y_test_pred = bag.predict(x_test)

bag_train = accuracy_score(y_train, y_train_pred)
bag_test = accuracy_score(y_test, y_test_pred)

print(f'Bagging train accuracy: {bag_train:.3f}')
print(f'Bagging test accuracy: {bag_test:.3f}')