# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #MEMpy - Introduction to Machine Learning

# <markdowncell>

# DTrees

# <codecell>

%pylab inline

# <codecell>

def plotDTrees(X,y, pair):        

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    
    n_classes = 3
    plot_colors = "bry"
    plot_step = 0.02
    
    # Standardize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    # Train
    clf = DecisionTreeClassifier().fit(X, y)
    
    # Plot the decision boundary
    #pl.subplot(2, 3, pairidx + 1)
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
    
    pl.xlabel(iris.feature_names[pair[0]])
    pl.ylabel(iris.feature_names[pair[1]])
    pl.axis("tight")
    
    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        pl.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i],
                   cmap=pl.cm.Paired)
    
    pl.axis("tight")
    pl.suptitle("Decision surface of a decision tree using paired features")
    pl.legend()
    pl.show()
    

# <codecell>

import numpy as np
import pylab as pl

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Parameters


# Load data
iris = load_iris()
X = iris.data[:, [1, 2]]
y = iris.target

plotDTrees(X, y, [1, 2])
    

# <codecell>

import numpy as np
import pylab as pl
from sklearn import cross_validation

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
clf = DecisionTreeClassifier()


X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.9, random_state=0)

clf = clf.fit(X_train, y_train)

print clf.score(X_test, y_test)

# <codecell>


