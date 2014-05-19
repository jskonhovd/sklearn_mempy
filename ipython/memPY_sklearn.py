# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #MEMpy - Introduction to Machine Learning

# <codecell>

%pylab inline

# <codecell>

def plotCustom(X,y, pair, Classifier, title="Custom Plot"):        

    # Shuffle
    idx = np.arange(X.shape[0])
    np.random.seed(13)
    np.random.shuffle(idx)
    #X = X[idx]
    #y = y[idx]
    
    n_classes = 3
    plot_colors = "bry"
    plot_step = 0.02
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    
    Z = Classifier.predict(np.c_[xx.ravel(), yy.ravel()])
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

def kmeans_plots(X, y, pair, Classifier, title="Custom plot"):
   
    idx = np.arange(X.shape[0])
    n_classes = 3
    plot_colors = "bry"
    plot_step = 0.02
    

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    h=.02
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
 
    # Obtain labels for each point in mesh. Use last trained model.
    
    # Put the result into a color plot
    Z = Classifier.predict(np.c_[xx.ravel(), yy.ravel()])
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
    centroids = Classifier.cluster_centers_
    pl.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w', zorder=10)
    pl.axis("tight")
    pl.suptitle(title)
    pl.legend()
    pl.show()

# <markdowncell>

# ## Decision Trees
# * Decision Tree learning is a method for approximating discrete-valued target functions, in which the learned function is represented a decision tree.
# * Maximize Information Gain
# 	* Information Gain measures how well a given attribute separates the training examples according to their target classification.

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
print clf.predict(X_test)
print y_test

# <codecell>

import numpy as np
import pylab as pl

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load data
iris = load_iris()
clf = DecisionTreeClassifier()
X = iris.data[:, [1, 2]]
y = iris.target
clf = clf.fit(X, y)
plotCustom(X, y, [1, 2], clf)
    

# <codecell>

import sys
from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

x_libsvm,y_libsvm = datasets.load_svmlight_file('spambase.libsvm')
X, y = shuffle(x_libsvm.todense(), y_libsvm)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

sizes = linspace(1, len(X_train), 20)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

for i, s in enumerate(sizes):
	clf = DecisionTreeClassifier(max_depth=3)
	clf.fit(X_train[:s], y_train[:s])

	train_err[i] = mean_squared_error(y_train[:s], clf.predict(X_train[:s]))
	test_err[i] = mean_squared_error(y_test, clf.predict(X_test))

pl.figure()
pl.title('DT: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('RMS Error')
pl.show()

# <markdowncell>

# ## kNN
# * K-Nearest neighbor algorithm
#     * kNN is a example of a instance based learning algorithm.
#     * Output is classified by a majority vote of its neighbors, where the class that is most common of a instances K neighbors.

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

clf = neighbors.KNeighborsClassifier(1, 'distance')
clf = clf.fit(X,y)
plotCustom(X, y, [1,2], clf)

# <codecell>

import sys
from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn import neighbors


x_libsvm,y_libsvm = datasets.load_svmlight_file('spambase.libsvm')
X, y = shuffle(x_libsvm.todense(), y_libsvm)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

sizes = linspace(1, len(X_train), 20)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

for i, s in enumerate(sizes):
	clf = neighbors.KNeighborsClassifier(10, 'distance')
	clf.fit(X_train[:s], y_train[:s])

	train_err[i] = mean_squared_error(y_train[:s], clf.predict(X_train[:s]))
	test_err[i] = mean_squared_error(y_test, clf.predict(X_test))

pl.figure()
pl.title('2-NN: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('RMS Error')
pl.show()


# <markdowncell>

# ## SVM
# * Support Vector Machines
#     *SVM's are a class of linear classifiers.
# * Kernel Trick

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
clf.fit(X,y)
rbf_svc.fit(X,y)
poly_svc.fit(X,y)
lin_svc.fit(X,y)
plotCustom(X, y, [1,2], rbf_svc)
plotCustom(X, y, [1,2], poly_svc)
plotCustom(X, y, [1,2], lin_svc)

# <codecell>

"""
Plots Learning curves for SVM
"""

import sys
from numpy import *
import pylab as pl
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.svm import SVR

x_libsvm,y_libsvm = datasets.load_svmlight_file('spambase.libsvm')
X, y = shuffle(x_libsvm.todense(), y_libsvm)
offset = int(0.7*len(X))
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

sizes = linspace(1, len(X_train), 20)
train_err = zeros(len(sizes))
test_err = zeros(len(sizes))

for i, s in enumerate(sizes):
	clf = SVR(kernel='rbf', degree=3)
	clf.fit(X_train[:s], y_train[:s])

	train_err[i] = mean_squared_error(y_train[:s], clf.predict(X_train[:s]))
	test_err[i] = mean_squared_error(y_test, clf.predict(X_test))

pl.figure()
pl.title('SVM: Performance vs Training Size')
pl.plot(sizes, test_err, lw=2, label = 'test error')
pl.plot(sizes, train_err, lw=2, label = 'training error')
pl.legend()
pl.xlabel('Training Size')
pl.ylabel('RMS Error')
pl.show()

# <markdowncell>

# ## KMeans
# * The k-means algorithm clusters data by trying to separate samples into n groups of equal variance.
# * The name is derived from the representing k clusters by the mean of its points.
# * K-Means works well with numerical attributes.

# <codecell>

from time import time
import numpy as np
import pylab as pl

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, [2, 3]]
y = iris.target
n_digits = len(np.unique(y))
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(X)

kmeans_plots(X,y,[2, 3],kmeans)

# <codecell>


