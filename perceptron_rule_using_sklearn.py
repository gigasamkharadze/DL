from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
x=iris.data[:, [2, 3]]
y=iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1, stratify=y)

sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

ppn = Perceptron(eta0=.1, random_state=1)
ppn.fit(x_train_std, y_train)

y_pred = ppn.predict(x_test_std)
print(f'Misclassified examples: {(y_test != y_pred).sum()}')
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

