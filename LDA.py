import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
x, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=0)

sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)

# find mean vectors
np.set_printoptions(precision=4)
mean_vectors = []
for label in range(1, 4):
    mean_vectors.append(np.mean(x_train_std[y_train == label], axis=0))

# compute the scaled within-class scatter matrix
d = 13  # number of features
s_w = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vectors):
    class_scatter = np.cov(x_train_std[y_train == label].T)
    s_w += class_scatter

# compute the between-class scatter matrix
mean_overall = np.mean(x_train_std, axis=0)
d = 13 
S_B = np.zeros((d, d))
for i, mean_vec in enumerate(mean_vectors):
    n = x_train[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d, 1)
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

# compute eigenvalues and eigenvectors of the matrix (S_W^-1 * S_B)
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(s_w).dot(S_B))
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# stack the two most discriminative eigenvector columns to create the transformation matrix W
W = np.hstack((eigen_pairs[0][1][:, np.newaxis].real, eigen_pairs[1][1][:, np.newaxis].real))

# transform the samples onto the new subspace
x_train_lda = x_train_std.dot(W)

# plot the transformed samples
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(x_train_lda[y_train == l, 0], x_train_lda[y_train == l, 1] * (-1), c=c, label=l, marker=m)

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()
