from autograd import numpy as np
from autograd import grad, jacobian
import numpy.matlib as nm
from svgd import SVGD
import sys
#from mpltools import style
#from mpltools import layout




num_particles = 50
#style.use('ggplot')

import matplotlib.pyplot as plt

#-(1.0/(2*observation_variance))*(theta_i  -  time_series[t])**2  + np.log(1.0/np.sqrt(np.pi*2*observation_variance))
observation_variance = .00000000001
transition_variance = 1000


class StateSpaceModel():
    def __init__(self):
      self.weights =[ ]
      self.grad_fn = grad(self.lnprob_theta_i)
    def lnprob_theta_i(self, theta_i, theta_t_minus_1, time_series,t,iter_):
        #ln poisson observations
            lnprob_theta_i = -np.exp(theta_i) + time_series[t]*theta_i - np.sum(np.log(np.arange(time_series[t])+1))
            if iter_ == 999:
                self.weights.append(lnprob_theta_i._value[0])
            transition_sum = 0
            for theta_t_minus_1_i in theta_t_minus_1:

                transition_sum += 1.0/(np.sqrt(2*np.pi*transition_variance))*np.exp(-.5*(1.0/transition_variance)*((theta_i - theta_t_minus_1_i )**2))
                
            return (lnprob_theta_i+np.log(transition_sum))
    
    def dlnprob(self, theta_i,theta_t_minus_1,time_series, t, iter_):
        return (self.grad_fn(theta_i, theta_t_minus_1, time_series,t , iter_))
    
    def grad_overall(self, theta,theta_t_minus_1,time_series, t, iter_):
	
	#from multiprocessing import Process, Manager

	#def f(d):
    #		d[1] += '1'
   # 		d['2'] += 2

    #	manager = Manager()

    #	d = manager.dict()

#	p1 = Process(target=f, args=(d,))
 #   	p2 = Process(target=f, args=(d,))
   # 	p1.start()
  #  	p2.start()
    #	p1.join()
    #	p2.join()

        return_matrix = []
	# we need to parallelize this to get realistic speeds
        for theta_i in theta:
            return_matrix.append(self.dlnprob(theta_i,theta_t_minus_1 ,time_series,t, iter_))
    
        return np.array(return_matrix)
    
if __name__ == '__main__':
    filtered_means = []
    filtered_covs = []
    total_thetas = []
    n_iter = 1000

    time_series = np.round(np.power(np.sin(np.arange(2)+1),2)*10 + 10)
    input_exists = False
    i = 1
    while input_exists:
        try:
            time_series.append(float(sys.argv[i].replace(",","")))
            i+=1
        except:
            input_exists =False


    model = StateSpaceModel()
    num_particles = 100
    x0 = np.random.normal(-10,1,[num_particles,1]).astype(float)
    weights = []
    theta = SVGD().update(x0,0,x0,time_series, model.grad_overall,n_iter=n_iter, stepsize=0.01)
    total_thetas.append(theta)
    #theta = p(x_0|y_0)

    
   
    filtered_means.append(np.mean(theta,axis=0)[0])
    filtered_covs.append(np.var(theta,axis=0)[0])
    
    for t in range(1,len(time_series)):
      print (t)
      theta = SVGD().update(theta,t,theta, time_series, model.grad_overall, n_iter=n_iter, stepsize=0.01)
      total_thetas.append(theta)
      filtered_means.append(np.mean(theta,axis=0)[0])
      filtered_covs.append(np.var(theta,axis=0)[0])
    #print (model.weights)
    return_list = filtered_means + filtered_covs + model.weights
    total_thetas = np.array(total_thetas).flatten()
    total_thetas = np.append(total_thetas,np.array(model.weights).flatten())
    myList = ','.join(map(str,total_thetas))
    print (myList)
   
