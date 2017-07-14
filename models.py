#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sktensor as skt
import scipy.linalg
from utils import rmse
from sklearn.base import BaseEstimator
import traceback

class TensorRegression(BaseEstimator):
    """
    Scikit learn estimator for various regression methods (see paper for details):
     - RLS:      Regularized Least Squares
     - LRR:      Low-rank regression (i.e. reduced rank regression)
     - HOLRR:    Higher-order low-rank regression
     - K-{ALGO}: Kernelized version of the above algorithms.
    """
    def __init__(self, algo, gamma=0., rank=None, kernel_fct=None, cov=None, beta=0.,noise_est=None):
        self.W = None
        assert algo in ['RLS','HOLRR', 'K_HOLRR', 'LRR', 'K_LRR', 'K_RLS'], 'algorithm %s not implemented' % algo
        self.algo = algo
        self.gamma = gamma
        self.rank = rank
        self.kernel_fct = kernel_fct
        self.cov = cov
        self.beta = beta
        self.noise_est = noise_est
        if cov is not None and np.linalg.matrix_rank(cov) < cov.shape[0]:
            cov += np.eye(cov.shape[0]) * 1e-5

    def get_params(self, deep=True):
        return {'gamma':self.gamma, 'rank':self.rank,
                'algo':self.algo, 'kernel_fct':self.kernel_fct,
                'beta':self.beta}

    def set_params(self,**d):
        for parameter, value in d.items():
            setattr(self,parameter, value)
        return self

    def predict(self,X):
        X = skt.dtensor(X)
        if self.algo[:2] == 'K_':
            X = X.unfold(0)
            K = self.kernel_fct.gram_matrix(self.traindata,X)
            return self.W.ttm(K,0)
        else:
            return self.W.ttm(X.unfold(0),0)

    def fit(self,X, Y):
        X = skt.dtensor(X)
        Y = skt.dtensor(Y)
        self.traindata = X
        try:
            if self.algo == "RLS":
                self.W = RLS(X,Y, gamma=self.gamma)
            elif self.algo == "HOLRR":
                self.W = HOLRR(X,Y,gamma=self.gamma,R=self.rank)
            elif self.algo == 'LRR':
                self.W = LRR(X,Y,rank=self.rank,gamma=self.gamma)
            elif self.algo == 'K_HOLRR':
                self.W = K_HOLRR(X,Y,rank=self.rank,kernel=self.kernel_fct,gamma=self.gamma)
            elif self.algo == 'K_RLS':
                self.W = K_RLS(X,Y,kernel=self.kernel_fct,gamma=self.gamma)
            elif self.algo == 'K_LRR':
                self.W = K_LRR(X,Y,self.rank,self.kernel_fct,self.gamma)
        except:
            traceback.print_exc()
            if self.algo[:2] == 'K_':
                self.W = skt.dtensor(np.zeros(X.shape[:1] + Y.shape[1:]))
            else:
                self.W = skt.dtensor(np.zeros(X.shape[1:] + Y.shape[1:]))

    def score(self,X,Y):
        return -1*self.loss(X,Y)

    def loss(self,X,Y):
        return rmse(self.predict(X),Y)


def RLS(X,Y, gamma = 0.):
    X_mat = X.unfold(0)
    Y_mat = Y.unfold(0)
    W_ols = np.linalg.inv(X_mat.T.dot(X_mat) + gamma * np.eye(X_mat.shape[1])).dot(X_mat.T).dot(Y_mat)
    W_ols.ten_shape = X_mat.shape[1:] + Y.shape[1:]
    return W_ols.fold()


def LRR(X,Y, rank, gamma = 0.):
    if type(X) == 'sktensor.dtensor.dtensor':
        X = X.unfold(0)

    Y_mat = Y.unfold(0)
    XtX = X.T.dot(X)
    XtX_inv = np.linalg.inv(XtX + gamma * np.eye(X.shape[1]))
    W_ols = skt.dtensor(XtX_inv.dot(X.T).dot(Y_mat)).unfold(0)
    W_ols.ten_shape = X.shape[1:] + Y.shape[1:]

    _,V = scipy.sparse.linalg.eigs(Y_mat.T.dot(X).dot(XtX_inv).dot(X.T).dot(Y_mat),k=rank)
    P = V.dot(V.T)
    W_rr = W_ols.dot(P).fold()


    return W_rr


def HOLRR(X,Y,R,gamma = 0.):
    if type(X) == 'sktensor.dtensor.dtensor':
        X = X.unfold(0)

    W_shape = X.shape[1:] + Y.shape[1:]
    I = np.eye(X.shape[1])
    M = []

    XX_inv = np.linalg.inv(X.T.dot(X) + gamma*I)
    A = XX_inv.dot(X.T).dot(Y.unfold(0)).dot(Y.unfold(0).T).dot(X)

    if R[0] == W_shape[0]:
        U = np.eye(R[0])
    else:
        ev,U = scipy.sparse.linalg.eigs(A,k=R[0])

    M.append(U)

    for i in range(1,len(R)):
        if R[i] == W_shape[i]:
            tmp = np.eye(W_shape[i])
        else:
            tmp = skt.core.nvecs(Y,i,R[i])
        M.append(tmp)

    U1 = M[0]
    G = Y.ttm([(np.linalg.inv(U1.T.dot(X.T.dot(X)+gamma*I).dot(U1)).dot(U1.T).dot(X.T)).T] + M[1:], transp=True)

    return G.ttm(M)


def K_HOLRR(X,Y,rank,kernel, gamma = 0., verbose = -1):
    K = kernel.gram_matrix(X)

    I = np.eye(X.shape[0])
    M = []
    W_shape = K.shape[1:] + Y.shape[1:]
    K_inv = np.linalg.inv(K + gamma*I)
    A = K_inv.dot(Y.unfold(0)).dot(Y.unfold(0).T).dot(K)
    if rank[0] == W_shape[0]:
        U = np.eye(rank[0])
    else:
        try:
            ev,U = scipy.sparse.linalg.eigs(A,k=rank[0])
        except scipy.sparse.linalg.ArpackNoConvergence:
            print "eigen decomposition did not converge... " + str(Y.shape) + " " + str(rank) + " " + str(kernel)
            return skt.dtensor(np.zeros(W_shape))
    M.append(U)

    for i in range(1,len(rank)):
        if rank[i] == W_shape[i]:
            tmp = np.eye(W_shape[i])
        else:
            tmp = skt.core.nvecs(Y,i,rank[i])
        M.append(tmp)

    U1 = M[0]
    G = Y.ttm([((np.linalg.inv(U1.T.dot(K).dot(K+gamma*I).dot(U1))).dot(U1.T).dot(K)).T] + M[1:], transp=True)

    return G.ttm(M)


def K_LRR(X,Y,R,kernel, gamma = 0.):
    K = kernel.gram_matrix(X)
    return LRR(K,Y,R,gamma)


def K_RLS(X,Y, kernel,gamma = 0.):
    K = kernel.gram_matrix(X)
    C = RLS(skt.dtensor(K),Y,gamma)
    return C

