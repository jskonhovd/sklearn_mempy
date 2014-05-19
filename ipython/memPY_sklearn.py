# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #MEMpy - Introduction to Machine Learning

# <markdowncell>

# Decision Trees

# <codecell>

%pylab inline

# <codecell>

def plotCustom(X,y, pair, Classifier, title="Custom Plot"):        

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
    clf = Classifier.fit(X, y)
    
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
    pl.suptitle(title)
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
clf = DecisionTreeClassifier()
X = iris.data[:, [1, 2]]
y = iris.target
clf = clf.fit(X, y)
plotCustom(X, y, [1, 2], clf)
    

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

# <markdowncell>

# KNN

# <codecell>

from sklearn import neighbors
import numpy as np
import pylab as pl
from sklearn import cross_validation
from sklearn.datasets import load_iris

iris = load_iris()

X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.9, random_state=0)

clf = neighbors.KNeighborsClassifier(3, 'distance')

clf = clf.fit(X_train, y_train)


print clf.score(X_test, y_test)

# <codecell>

from sklearn import neighbors
import numpy as np
import pylab as pl
from sklearn import cross_validation
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, [1, 2]]
y = iris.target

clf = neighbors.KNeighborsClassifier(20, 'distance')

plotCustom(X, y, [1,2], clf)

# <markdowncell>

# SVM

# <codecell>

from sklearn import svm
import numpy as np
import pylab as pl
from sklearn import cross_validation
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, [1, 2]]
y = iris.target
C = 1.0
clf = svm.SVC(kernel='linear', C=C)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C)
lin_svc = svm.LinearSVC(C=C)

plotCustom(X, y, [1,2], rbf_svc)
plotCustom(X, y, [1,2], poly_svc)
plotCustom(X, y, [1,2], lin_svc)

# <markdowncell>

# KMeans

# <codecell>


