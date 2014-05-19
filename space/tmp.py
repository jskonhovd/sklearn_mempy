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

n_digits = len(np.unique(y_iris))
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=100)
kmeans.fit(xd_iris)

import sys

def kmeans_plots(km,xd,y, h=.02):
   
    reduced_data = xd
    #reduced_data = PCA(n_components=2).fit_transform(xd)
    #km.fit(reduced_data)
    x_min, x_max = reduced_data[:, 0].min() , xd[:, 0].max() 
    y_min, y_max = reduced_data[:, 1].min() , xd[:, 1].max() 
    

    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
 
    # Obtain labels for each point in mesh. Use last trained model.
    Z = km.predict(np.c_[xx.ravel(), yy.ravel()])
    
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure(1)
    pl.clf()
    pl.imshow(Z, interpolation='nearest',
          extent=(xx.min(), xx.max(), yy.min(), yy.max()),
          cmap=pl.cm.Paired,
          aspect='auto', origin='lower')

    pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    pl.scatter(centroids[:, 0], centroids[:, 1],
           marker='x', s=169, linewidths=3,
           color='w', zorder=10)
    pl.title('K-means clustering on the iris dataset (PCA-reduced data)\n'
         'Centroids are marked with white cross')
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.xticks(())
    pl.yticks(())
    pl.show()


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
