---
title: MEMpy - A Introduction to Machine Learning
author: Jeffrey Skonhovd
email: jskonhovd@gatech.edu
---


# Introduction

## Who am I?
* Jeffrey Skonhovd
* Works at FTN Financial
* Twitter: @jskonhovd
* Github: jskonhovd

## Overview
* What is Machine Learning?
  * Machine Learning is the study of computer algorithms that improve automatically through experience.
* How should I go about learning Machine Learning?
  * MOOCs
  * Don't get caught up in the implementations. 
* Tools
  * WEKA
  * scikit-learn

# Machine Learning
## Types
* Supervised Learning
  * Supervised Learning is ...
* Unsupervised Learning
  * Unsupervised Learning is ...
* Reenforcement Learning
  * Reenforcement Learning is ...
  
## Some Boring, but important Definitions.
* Inductive Bias
  * The inductive bias of a learning algorithm is the set of assumptions that the learner uses to predict outputs given inputs that it has not encountered.
  * Occam's Razor assumes that the hypotheses with the fewest assumptions should be selected.
* Cross-validation
  * The basic idea of Cross-validation to leave out some of the data when fitting the model.

## Scikit-learn
* Scikit-learn is a set of simple and efficient tools for data mining and data analysis.
* Uses Python!!!
* [http://scikit-learn.org/](http://scikit-learn.org/)

# Supervised Learning: Scikit-learn

## Decision Trees
* Decision Tree learning is a method for approximating discrete-valued target functions, in which the learned function is represented a decision tree.
* Maximize Information Gain
  * Information Gain measures how well a given attribute separates the training examples according to their target classifcation.

## Decision Trees: Example
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
    plotCustom(X, y, [1, 2], clf)`

## kNN
  
## SVM



# Unsupervised Learning: Scikit-learn
## kMeans

## PCA

# Conclusion

* Resources
