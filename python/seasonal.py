from autograd import numpy as np
from autograd import grad, jacobian,elementwise_grad
import numpy.matlib as nm
from svgd import SVGD
import sys
#from mpltools import style
#from mpltools import layout




#style.use('ggplot')

import matplotlib.pyplot as plt

#-(1.0/(2*observation_variance))*(theta_i  -  time_series[t])**2  + np.log(1.0/np.sqrt(np.pi*2*observation_variance))
observation_variance = .00000000001
transition_variance = 1000
seasonality = 6

G = np.matrix([[np.cos(2*np.pi/seasonality),np.sin(2*np.pi/seasonality)],[-np.sin(2*np.pi/seasonality),np.cos(2*np.pi/seasonality)]])

class StateSpaceModel:

    def lnprob_theta_i(self, theta_i, theta_t_minus_1, time_series,t):
        #ln poisson observations
            lnprob_theta_i = -np.exp(theta_i[0]) + time_series[t]*theta_i[0] - np.sum(np.log(np.arange(time_series[t])+1))
            transition_sum = 0
            for theta_t_minus_1_i in theta_t_minus_1:
                tmp = np.transpose(np.matmul(G,theta_t_minus_1_i.reshape((-1,1)))).tolist()[0]
              
                transition_sum += 1.0/(np.sqrt(2*np.pi*transition_variance))*np.exp(-.5*(1.0/transition_variance)*((theta_i - tmp )**2))
            
            return (lnprob_theta_i+np.log(transition_sum))
    
    def dlnprob(self, theta_i,theta_t_minus_1,time_series, t):
        return (elementwise_grad(self.lnprob_theta_i)(theta_i, theta_t_minus_1, time_series,t))
    
    def grad_overall(self, theta,theta_t_minus_1,time_series, t, iter_):
        return_matrix = []

        for theta_i in theta:
            return_matrix.append(self.dlnprob(theta_i,theta_t_minus_1 ,time_series,t))
    
        return np.array(return_matrix)
    
if __name__ == '__main__':
    filtered_means = []
    filtered_covs = []
    total_thetas = []
    n_iter = 1000

    time_series = np.round(np.power(np.sin(np.arange(10)+1),2)*10 + 10)

   
    model = StateSpaceModel()
    num_particles = 10
    x0 = np.random.normal(0,10,[num_particles,2]).astype(float)
    
    theta = SVGD().update(x0,0,x0,time_series, model.grad_overall, n_iter=n_iter, stepsize=0.01)
    total_thetas.append(theta)
    #theta = p(x_0|y_0)

    
   
    filtered_means.append(np.mean(theta,axis=0)[0])
    filtered_covs.append(np.var(theta,axis=0)[0])
    
    for t in range(1,len(time_series)):
      theta = SVGD().update(theta,t,theta, time_series, model.grad_overall, n_iter=n_iter, stepsize=0.01)
      total_thetas.append(theta)
      filtered_means.append(np.mean(theta,axis=0)[0])
      filtered_covs.append(np.var(theta,axis=0)[0])
    
    return_list = filtered_means + filtered_covs
    myList = ','.join(map(str,np.array(total_thetas).flatten() ))
    print (myList)
   
