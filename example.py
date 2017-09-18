


"""
Small experiments on synthetic data
"""

from synthetic_data import linear_data_tucker,poly_data_tucker
import itertools
import models
from sklearn.model_selection import GridSearchCV
import utils


for xp in ['linear', 'kernel']:
    print '\n' * 3
    print "______________________________________________\n"
    print "Comparison of %s models on synthetic data." % xp
    print "______________________________________________"

    if xp == 'linear':
        X,Y,X_test,Y_test,_ = linear_data_tucker(dim_in=[10],
                                                 dim_out=[10,10],
                                                 ranks=[4,6,2],
                                                 N_train=20,
                                                 N_test=500,
                                                 noise=0.1)
    else:
        X, Y, X_test, Y_test, _  =  poly_data_tucker(dim_in=[10],
                                                     dim_out=[10, 10],
                                                     ranks=[4, 6, 2],
                                                     N_train=50,
                                                     N_test=500,
                                                     noise=0.1)

    # Grids for hyperparamater tuning with cross validation
    CV_tensor_ranks = [idx for idx in itertools.product([2,4,6],[2,4,6],[2,4,6])]
    CV_matrix_rank  = [2,4,6,8]
    CV_gamma = [10**n for n in range(-5,3)]
    CV_kernels = [utils.Kernel('poly', (2, 0)), utils.Kernel('poly', (2, 1)), utils.Kernel('poly', (4, 0))]

    res = ''

    for algo in ['RLS','LRR','HOLRR']:
        if xp == 'kernel':
            algo = 'K_' + algo

        print
        print "Cross-validation for",algo
        mdl = models.TensorRegression(algo)

        params = {}
        params['rank'] =  CV_tensor_ranks if 'HOLRR' in algo else CV_matrix_rank
        params['gamma'] = CV_gamma
        params['kernel_fct'] = CV_kernels

        # Cross validation
        gs = GridSearchCV(mdl, params, refit=False, verbose=True)
        gs.fit(X, Y)

        # get best model
        best_estimator = models.TensorRegression(algo)
        best_estimator.set_params(**gs.best_params_)
        best_estimator.fit(X, Y)

        # display results
        if 'RLS' not in algo:
            res += '     %s\trmse = %f\trank: %s, gamma: %f' % (algo,best_estimator.loss(X_test,Y_test), str(best_estimator.rank), best_estimator.gamma)
        else:
            res += '     %s\trmse = %f\tgamma: %f' % (algo,best_estimator.loss(X_test,Y_test), best_estimator.gamma)

        if 'K_' in algo:
            res += ' kernel: %s' % best_estimator.kernel_fct
        res += '\n'

    print '\n\n                 **  Results  **\n'
    print res





