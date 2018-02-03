import numpy as np
from scipy.spatial.distance import pdist, squareform
weights = []
class SVGD():

    def __init__(self):
        pass
    
    def svgd_kernel(self, theta, h = -1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            
            h = np.sqrt(0.5 * h / np.log(theta.shape[0]+1))

        # compute the rbf kernel
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
 
    def update(self, x0, t, theta_t_minus_1,time_series, dlnprob, n_iter = 10, stepsize = 1e-3, bandwidth = -1, alpha = 0.9, debug = False):
        # Check input
        if x0 is None or dlnprob is None:
            raise ValueError('x0 or dlnprob cannot be None!')
        
        theta = np.copy(x0) 
        
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        for iter in range(n_iter):
            if debug and (iter+1) % 1000 == 0:
                #print 'iter ' + str(iter+1) 
                pass
            lnpgrad = dlnprob(theta,theta_t_minus_1,time_series,t, iter)
            # calculating the kernel matrix
           # h = 0
           
            kxy, dxkxy = self.svgd_kernel(theta, h = -1)  
            grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]  
            
            # adagrad 
            if iter == 0:
                historical_grad = historical_grad + grad_theta ** 2
            else:
                historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
            adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
            theta = theta + stepsize * adj_grad 
            
        return theta


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

class StateSpaceModel():
    def __init__(self):
      self.grad_fn = grad(self.lnprob_theta_i)
    def lnprob_theta_i(self, theta_i, theta_t_minus_1, time_series,t,iter_):
        #ln poisson observations
            lnprob_theta_i = 1.0/(np.sqrt(2*np.pi*observation_variance))*np.exp(-.5*(1.0/observation_variance)*((time_series[t] - theta_i )**2))

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

	# we need to parallelize this to get realistic speeds
    	theta_split = np.split(theta,len(theta)/10)
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

	if iter_ == 4999:
		weights_tmp = []
		for theta_i in theta:
			weights_tmp.append(np.log(1.0/(np.sqrt(2*np.pi*observation_variance))) + -.5*(1.0/observation_variance)*((time_series[t] - theta_i )**2))
		weights.append(weights_tmp)
	return np.array(return_matrix)
    
if __name__ == '__main__':
    filtered_means = []
    filtered_covs = []
    total_thetas = []
    n_iter = 5000

    time_series = []#np.round(np.power(np.sin(np.arange(2)+1),2)*10 + 10)
    input_exists = True
    i = 1
    while input_exists:
        try:
            time_series.append(float(sys.argv[i].replace(",","")))
            i+=1
        except:
            input_exists =False


    model = StateSpaceModel()
    num_particles = 1000
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
   
    total_thetas = np.array(total_thetas).flatten()
    total_thetas = np.append(total_thetas,np.array(weights).flatten())
    weights = np.array(weights)
    myList = ','.join(map(str,total_thetas))
    print (myList)




    
