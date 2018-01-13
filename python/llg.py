from autograd import numpy as np
from autograd import grad, jacobian
import numpy.matlib as nm
from svgd import SVGD
import sys
#from mpltools import style
#from mpltools import layout

from multiprocessing import Process, Manager


#style.use('ggplot')

import matplotlib.pyplot as plt

#-(1.0/(2*observation_variance))*(theta_i  -  time_series[t])**2  + np.log(1.0/np.sqrt(np.pi*2*observation_variance))
observation_variance = 1
transition_variance = 10
weights = []

class StateSpaceModel():
    def __init__(self):
      self.weights =[ ]
      self.grad_fn = grad(self.lnprob_theta_i)
    def lnprob_theta_i(self, theta_i, theta_t_minus_1, time_series,t,iter_):
        #ln poisson observations
            lnprob_theta_i = 1.0/(np.sqrt(2*np.pi*observation_variance))*np.exp(-.5*(1.0/observation_variance)*((time_series[t] - theta_i )**2))

            if iter_ == 199:
		print ("hello")
                weights.append(lnprob_theta_i._value[0])
            transition_sum = 0
            for theta_t_minus_1_i in theta_t_minus_1:

                transition_sum += 1.0/(np.sqrt(2*np.pi*transition_variance))*np.exp(-.5*(1.0/transition_variance)*((theta_i - theta_t_minus_1_i )**2))
                
            return (np.log(lnprob_theta_i)+np.log(transition_sum))
    
    def dlnprob(self, theta_i,theta_t_minus_1,time_series, t, iter_):
        return (self.grad_fn(theta_i, theta_t_minus_1, time_series,t , iter_))
    
    def grad_overall(self, theta,theta_t_minus_1,time_series, t, iter_):
	

	def f(d,b,theta_b,theta_t_minus_1,time_series,t,iter_):
		return_matrix = []
		for theta_i in theta_b:
            		return_matrix.append(self.dlnprob(theta_i,theta_t_minus_1 ,time_series,t, iter_))

		d[b] = return_matrix
    	manager = Manager()

    	d = manager.dict()
        jobs = []
#	p1 = Process(target=f, args=(d,))
 #   	p2 = Process(target=f, args=(d,))
   # 	p1.start()
  #  	p2.start()
    #	p1.join()
    #	p2.join()

	# we need to parallelize this to get realistic speeds
    	theta_split = np.split(theta,len(theta)/5)
	for i in range(len(theta_split)):
		p = Process(target=f, args=(d,i,theta_split[i],theta_t_minus_1,time_series,t,iter_))
		jobs.append(p)
	for job in jobs:
		job.start()
	for job in jobs:
		job.join()
        return_matrix = []
	keylist = d.keys()
	keylist.sort()
	for key in keylist:
    		return_matrix += d[key]

	return np.array(return_matrix)
    
if __name__ == '__main__':
    filtered_means = []
    filtered_covs = []
    total_thetas = []
    n_iter = 200

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
    x0 = np.random.normal(-2,1,[num_particles,1]).astype(float)
    weights = []
    svgd = SVGD()
    theta = svgd.update(x0,0,x0,time_series, model.grad_overall,n_iter=n_iter, stepsize=0.01)
    total_thetas.append(theta)
    #theta = p(x_0|y_0)

    
   
    filtered_means.append(np.mean(theta,axis=0)[0])
    filtered_covs.append(np.var(theta,axis=0)[0])
    
    for t in range(1,len(time_series)):
      svgd = SVGD()
      theta = svgd.update(theta,t,theta, time_series, model.grad_overall, n_iter=n_iter, stepsize=0.01)
      total_thetas.append(theta)
      filtered_means.append(np.mean(theta,axis=0)[0])
      filtered_covs.append(np.var(theta,axis=0)[0])
    print (weights) 
    print (np.array(weights).shape) 
    print (np.array(total_thetas).shape)
    return_list = filtered_means + filtered_covs + model.weights
    total_thetas = np.array(total_thetas).flatten()
    total_thetas = np.append(total_thetas,np.array(weights).flatten())
    myList = ','.join(map(str,total_thetas))
    print (myList)
    print (total_thetas.shape)
