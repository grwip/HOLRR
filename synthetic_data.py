#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sktensor import *
import sktensor as skt
import numpy as np

def poly_data_tucker(dim_in, dim_out, ranks, N_train, N_test, noise=0):
    d_in = np.prod(dim_in)
    U = [np.random.normal(0,1,[d, R]) for d,R in zip([d**2 for d in dim_in] + dim_out,ranks)]
    G = dtensor(np.random.normal(0,1, ranks))
    W = G.ttm(U)
    X = skt.dtensor(np.random.normal(0,1,[N_train, d_in]))
    X_test = skt.dtensor(np.random.normal(0,1,[N_test, d_in]))

    X_poly = skt.khatrirao((X.T,X.T)).T
    X_test_poly = skt.khatrirao((X_test.T,X_test.T)).T
    Y = W.ttm(X_poly,0)
    Y_test = W.ttm(X_test_poly,0)

    z = Y.std()
    Y = Y / z
    Y_test = Y_test / z
    if noise > 0:
        Y += np.random.normal(0,noise, Y.shape)
        Y_test += np.random.normal(0,noise, Y_test.shape)

    return (X,Y,X_test,Y_test,W)

def linear_data_tucker(dim_in, dim_out, ranks, N_train, N_test, noise=0):
    d_in = np.prod(dim_in)
    U = [np.random.normal(0,1,[d, R]) for d,R in zip(dim_in + dim_out,ranks)]
    G = dtensor(np.random.normal(0,1, ranks))
    W = G.ttm(U)

    X = skt.dtensor(np.random.normal(0,1,[N_train, d_in]))
    X_test = skt.dtensor(np.random.normal(0,1,[N_test, d_in]))

    Y = W.ttm(X,0)
    Y_test = W.ttm(X_test,0)
    if noise > 0:
        Y += np.random.normal(0,noise, Y.shape)
        Y_test += np.random.normal(0,noise, Y_test.shape)


    return (X,Y,X_test,Y_test,W)
