from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x=iris.data[:, [2, 3]]
y=iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=1, stratify=y)
tree_model = DecisionTreeClassifier(
    criterion='gini', 
    max_depth=4, 
    random_state=1)

tree_model.fit(x_train, y_train)
tree.plot_tree(tree_model)
plt.show()