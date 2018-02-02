#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:05:11 2018

@author: gcgibson
"""
import sys
import numpy as np
import numpy.matlib as nm
from svgd_original import SVGD

def dln_gaussian_mean(x,mean,var):
    return  (1.0/var)*(x - mean )

def dln_binom(x,mean):
    ##dbinom x-mean (mean,1-e^-t)
    return np.log(1-np.pow(np.e,-.5)) - np.log(np.pow(np.e,-.5))


def dln_observation_density(x,mean):
    return dln_gaussian_mean(x,mean,1)

def dln_transition_density(x,mean):
    return dln_binom(x,mean)

def dln_prior(x):
    return dln_gaussian_mean(x,0,1)
class SSM:
    def __init__(self, time_series):
        self.time_series = time_series
        
    def dlnprob(self, theta):
        lambda_ = 3
        
        theta = theta.reshape((300,3,20))
        theta_new = []
        for theta_i in theta:
            tmp_p = []
            tmp_p.append(dln_prior(theta_i[0])+dln_observation_density(self.time_series[0],theta_i[0]))
            
            for i in range(1,len(self.time_series)):
                tmp_p.append(dln_observation_density(self.time_series[i],theta_i[i])  + dln_transition_density(theta_i[i],theta_i[i-1]))
            theta_new.append(tmp_p)
        
        
        return theta_new
    
if __name__ == '__main__':
    #time_series =np.round(np.power(np.sin(np.arange(100)+1),2)*10 + 10)
    with open("/Users/gcgibson/Stein-Variational-Gradient-Descent/python/dat.json") as f:
        dat = f.read()
    
    
    dat = dat.split(",")
    time = []
    cases = []
    count = 0
    for elm in dat:
        if count % 2 ==0:
            time.append(elm.split(":")[1])
        else:
            cases.append(int(elm.split(":")[1].replace("}","").replace(']"]\n',"")))
        count +=1
    
    
    time_series = np.array(cases)
    time_series = time_series[:20]
    from timeit import default_timer as timer
    start = timer()

    model = SSM(time_series)
    
    x0 = np.random.multivariate_normal(np.zeros(len(time_series)),np.eye(len(time_series)),900);
    theta = SVGD().update(x0, model.dlnprob, n_iter=1000, stepsize=1)
    import matplotlib.pyplot as plt
    end = timer()
    print(end - start)  
    plt.plot(range(len(time_series)), np.mean(theta,axis=0))
    plt.plot(range(len(time_series)), time_series,color='b')
    plt.fill_between(range(len(time_series)),np.percentile(theta,5,axis=0),np.percentile(theta,95,axis=0),alpha=.3)
    plt.show()