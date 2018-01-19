#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:05:11 2018

@author: gcgibson
"""

import numpy as np
import numpy.matlib as nm
from svgd_original import SVGD

def dln_gaussian_mean(x,mean,var):
    return  (1.0/var)*(x - mean )


def dln_observation_density(x,mean):
    return dln_gaussian_mean(x,mean,1)

def dln_transition_density(x,mean):
    return dln_gaussian_mean(x,mean,10)

def dln_prior(x):
    return dln_gaussian_mean(x,0,1)
class SSM:
    def __init__(self, time_series):
        self.time_series = time_series
        
    def dlnprob(self, theta):
        theta_new = []
        for theta_i in theta:
            tmp_p = []
            tmp_p.append(dln_prior(theta_i[0])+dln_observation_density(self.time_series[0],theta_i[0]))
            for i in range(1,len(self.time_series)):
                tmp_p.append(dln_observation_density(self.time_series[i],theta_i[i])  + dln_transition_density(theta_i[i],theta_i[i-1]))
            theta_new.append(tmp_p)
        return theta_new
    
if __name__ == '__main__':
    time_series =np.round(np.power(np.sin(np.arange(100)+1),2)*10 + 10)
    
    model = SSM(time_series)
    
    x0 = np.random.multivariate_normal(np.zeros(len(time_series)),np.eye(len(time_series)),100);
    theta = SVGD().update(x0, model.dlnprob, n_iter=2000, stepsize=0.01)
    import matplotlib.pyplot as plt
    
    plt.plot(range(len(time_series)), np.mean(theta,axis=0))
    plt.plot(range(len(time_series)), time_series)
    plt.fill_between(range(len(time_series)),np.percentile(theta,5,axis=0),np.percentile(theta,95,axis=0),alpha=.3)
    plt.show()