##%pylab inline
##%matplotlib gtk
##%matplotlib inline

from sklearn import datasets
import numpy as np
import pylab as pl

##Kmeans
n_digits = len(np.unique(y_iris))
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=100)
kmeans.fit(xd_iris)

##PCA

pca_iris = PCA(n_components=2).fit(xd_iris)
