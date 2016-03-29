__author__ = 'Mind'
import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.metrics as skmet
import sklearn.grid_search as skgs
import sklearn.ensemble as ske
from sklearn.ensemble import ExtraTreesClassifier


#feature selection procedure
def feature_selection(X_train, Y):
    """
    input:
    X_train: training set
    Y: labels
    output:
    Xtrain: matrix containing the highest scoring features
    """
    print(X_train.shape)
    print(X_train[:1])
    clf = ExtraTreesClassifier()
    X_train = clf.fit(X_train, Y).transform(X)
    print(clf.feature_importances_)
    print(X_train.shape)
    print(X_train[1])
    return X_train


# score function
def classification_score(Y_pred, Y_true):
    """
    input:
    Ypred: matrix with predicted values
    Ytrue: matrix with actual values
    output:
    accuracy score
    """
    score = skmet.accuracy_score(Y_true, Y_pred)
    return score


# feature scale
def feature_scale(X):
    """
    input:
    X: training/validation matrix to be normalized
    output:
    X: normalized matrix (no normalization is applied to the categorical data)
    """
    X_cate = X[:, 9:]
    X_scale = preprocessing.scale(X[:, 0:9])
    X = np.concatenate((X_scale, X_cate), axis=1)
    return X


# apply transformation to the data
def transformX(X):
    """
    input:
    X: training/validation matrix
    output:
    X: transformed (added/removed) columns matrix
    """
    Apower2 = X[:, 0] ** 2
    Ipower2 = X[:, 8] ** 2
    Epower2 = X[:, 4] ** 2

    AIscalar = X[:, 0] * X[:, 8]
    AEscalar = X[:, 0] * X[:, 4]
    IEscalar = X[:, 8] * X[:, 4]
    AIEplus = X[:, 0] + X[:, 8] + X[:, 4]
    AIEscalar = X[:, 0] * X[:, 8] * X[:, 4]
    X = np.column_stack(
        (X, Apower2, Ipower2, Epower2,
         AIEplus, AIEscalar
         , AIscalar, AEscalar, IEscalar))

    # remove A,B,H
    X = np.delete(X, [0, 1, 7], axis=1)
    return X


# extra trees implementation + cross validation
def extra_trees(X_train, Y_train, X_test):
    """
    input:
    X_train: feature scaled and normalized matrix - training set
    Y_train: training labels matrix
    X_test: feature scaled and normalized matrix - validation set
    output:
    Y_pred: matrix with predictions over the validation set
    """
    # split Y into 2
    Y1 = Y_train[:, 0]
    Y2 = Y_train[:, 1]
    # extra trees classifier
    clf = ske.ExtraTreesClassifier()
    # cross validation
    scorefun = skmet.make_scorer(classification_score)
    # param_grid = {}
    param_grid = {'max_features': ('auto', 'sqrt', 'log2'), 'n_estimators': [50, 150],
                  'min_samples_split': [2, 8]}
    grid_search = skgs.GridSearchCV(clf, param_grid, scoring=scorefun, cv=5)
    # train specifically for Y1
    grid_search.fit(X_train, Y1)
    score_Y1 = grid_search.best_score_
    best_Y1 = grid_search.best_estimator_
    # predict based on the best estimator for Y1
    Y_pred1 = best_Y1.predict(X_test)
    # train specifically for Y2
    grid_search.fit(X_train, Y2)
    score_Y2 = grid_search.best_score_

    # overall cross validation score
    cv_score = 1 - ((score_Y1 + score_Y2) / 2)
    print('C-V score =', cv_score)
    best_Y2 = grid_search.best_estimator_
    # predict based on the best estimator for Y2
    Y_pred2 = best_Y2.predict(X_test)

    # concatenate the individual Y1 and Y2 predictions
    Y_pred = np.column_stack((Y_pred1, Y_pred2))
    return Y_pred

# pandas labels
names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K+1', 'K+2', 'K+3', 'K+4', 'L+1', 'L+2', 'L+3',
         'L+4', 'L+5', 'L+6', 'L+7', 'L+8', 'L+9', 'L+10', 'L+11', 'L+12', 'L+13', 'L+14', 'L+15', 'L+16',
         'L+17',
         'L+18', 'L+19', 'L+20', 'L+21', 'L+22', 'L+23', 'L+24', 'L+25', 'L+26', 'L+27', 'L+28', 'L+29',
         'L+30', 'L+31', 'L+32', 'L+33', 'L+34', 'L+35', 'L+36', 'L+37', 'L+38', 'L+39', 'L+40']
# read the data
X = pd.read_csv('train.csv', sep=',', names=names, dtype=float)
X = X.as_matrix()
Y = pd.read_csv('train_y.csv', names=['Y1', 'Y2'])
Y = Y.as_matrix()

X_valid = pd.read_csv('test.csv', sep=',', names=names, dtype=float)
X_valid = X_valid.as_matrix()

# feature scale
X = feature_scale(X)
X_valid = feature_scale(X_valid)

# transformations
X = transformX(X)
X_valid = transformX(X_valid)

# cross validation + extra trees
Y_pred = extra_trees(X, Y, X_valid)

# output to csv
Y_pred = pd.DataFrame(data=Y_pred, dtype=int)
Y_pred.to_csv('out101.csv', index=False, header=False)


