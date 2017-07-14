#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sktensor as skt

def rmse(Y1,Y2):
    return np.sqrt(np.linalg.norm(Y1-Y2)**2/Y2.size)

class Kernel(object):
    def __init__(self,name,param=None, normalize=False):
        assert name in ['rbf','linear','poly'], 'kernel not implemented'
        if name == 'rbf':
            self.f = lambda x,y: np.exp(-1*param * np.linalg.norm(x-y)**2)
            self.param = param
            self.name = name

        if name == 'linear':
            self.f = np.dot
            self.param = param
            self.name = name

        if name == 'poly':
            self.f = lambda x,y: (param[1] + x.T.dot(y))**param[0]
            self.param = param
            self.name = name
        self.normalize = normalize

    def __call__(self,x,y):
        if self.normalize:
            z = np.sqrt(self.f(x,x)*self.f(y,y))
        else:
            z = 1
        return self.f(x,y) / z

    def __str__(self):
        return self.name + ' (param=%s, norm=%s)' % (str(self.param), str(self.normalize))

    def gram_matrix(self,X,X_test=None):
        if isinstance(X,skt.dtensor):
            X = X.unfold(0)
        if X_test is None:
            n = X.shape[0]
            K = np.zeros([n,n])
            for i in range(n):
                for j in range(i,n):
                    tmp = self(X[i,:],X[j,:])
                    K[i,j] = tmp
                    K[j,i] = tmp
        else:
            if isinstance(X_test,skt.dtensor):
                X_test = X_test.unfold(0)
            K = np.zeros([X_test.shape[0], X.shape[0]])
            for i in range(X_test.shape[0]):
                for j in range(X.shape[0]):
                    K[i,j] = self(X_test[i,:],X[j,:])
        return K




def rbf_kernel(gamma):
    return lambda x,y: np.exp(-1*gamma * np.linalg.norm(x-y)**2)
