#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 19:17:15 2018

@author: gcgibson
"""

from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc.optimizers import adam


import scipy

def black_box_variational_inference(time_series,logprob, D, num_samples):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        # Variational dist is a diagonal Gaussian.
        mean, log_std = params[:D], params[D:]
        return mean, log_std

    def gaussian_entropy(log_std):
        return 0.5 * D * (1.0 + np.log(2*np.pi)) + np.sum(log_std)

    rs = npr.RandomState(0)
    def variational_objective(params, t):
        """Provides a stochastic estimate of the variational lower bound."""
        mean, log_std = unpack_params(params)
        samples = rs.randn(num_samples, D) * np.exp(log_std) + mean
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(time_series,samples, t))
        return -lower_bound

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params


def binom_logpmf(x,n,p):
    return x*np.log(p) + (n-x)*np.log(1-p)

if __name__ == '__main__':

    # Specify an inference problem by its unnormalized log-density.
    # full SSM likelihood decomposition is given by
    #  lnp(X_1) +  \sum_{i=1}^n lnp(y_i |x_i) +  \sum_{2}^n lnp(x_i | x_{i-1})
    time_series = np.random.normal(100,1,10)
    rho = .8
    D = 4*3
    def log_density(time_series,x, t):
        S, I, R = x[:, 0:D/3], x[:, D/3:2*D/3], x[:, 2*D/3:D]
        lnprob_vector =[]
        for sample_index in range(S.shape[0]):
            lnprob = 0
            for time_series_index in range(1,S.shape[1]):
                
                #lnprob +=  norm.logpdf(time_series[time_series_index], S[sample_index][time_series_index], 10)
                #lnprob +=  norm.logpdf(S[sample_index][time_series_index], S[sample_index][time_series_index-1], .1)
                delta_n_si = S[sample_index][time_series_index] - I[sample_index][time_series_index]
                delta_n_ir = I[sample_index][time_series_index] - R[sample_index][time_series_index]
                
                lnprob += binom_logpmf(delta_n_si,S[sample_index][time_series_index],rho)
                lnprob += binom_logpmf(delta_n_ir,I[sample_index][time_series_index],rho)
                lnprob += binom_logpmf(time_series[time_series_index],I[sample_index][time_series_index],rho)
            lnprob_vector.append(lnprob)
        lnprob_vector = np.array([0] + lnprob_vector)
        print (t)
        return lnprob_vector
    
    # Build variational objective.
    objective, gradient, unpack_params = \
        black_box_variational_inference(time_series,log_density, D, num_samples=100)

    # Set up plotting code
   

    print("Optimizing variational parameters...")
    init_mean    = -1 * np.ones(D)
    init_log_std = -5 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=2000)
    