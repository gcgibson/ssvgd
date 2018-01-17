import numpy as np
from scipy.spatial.distance import pdist, squareform
weights = []


import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

from autograd import grad
from autograd.misc.optimizers import adam


def black_box_variational_inference(logprob, D, last_theta, time_series, current_t, num_samples):
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
        lower_bound = gaussian_entropy(log_std) + np.mean(logprob(samples,last_theta,time_series,current_t, t))
        return -lower_bound
    
    gradient = grad(variational_objective)
    
    return variational_objective, gradient, unpack_params


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
    def lnprob_theta_i(self, theta_i, theta_t_minus_1, time_series,current_t,t):
        #ln poisson observations
            lnprob_theta_i = 1.0/(np.sqrt(2*np.pi*observation_variance))*np.exp(-.5*(1.0/observation_variance)*((time_series[current_t] - theta_i )**2))

            transition_sum = 0
            for theta_t_minus_1_i in theta_t_minus_1:

                transition_sum += 1.0/(np.sqrt(2*np.pi*transition_variance))*np.exp(-.5*(1.0/transition_variance)*((theta_i - theta_t_minus_1_i )**2))
                
            return (np.log(lnprob_theta_i)+np.log(transition_sum))
    

    
if __name__ == '__main__':
    filtered_means = []
    filtered_covs = []
    total_thetas = []
    

    time_series = np.round(np.power(np.sin(np.arange(10)+1),2)*10 + 10)
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
    weights = [x0]
    D = 1
    objective, gradient, unpack_params = \
        black_box_variational_inference(model.lnprob_theta_i, D,x0 , time_series, 0, num_samples=200)
    init_mean    = -1 * np.ones(D)
    init_log_std = -5 * np.ones(D)
    init_var_params = np.concatenate([init_mean, init_log_std])
    variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=500)
    total_thetas.append(variational_params.tolist())
    #theta = p(x_0|y_0)
    
    

    for t in range(1,len(time_series)):
        x0 = total_thetas[len(total_thetas) - 1]
        x0 = np.random.normal(x0[0],np.exp(x0[1]),[num_particles,1])
        weights.append(x0)
        objective, gradient, unpack_params = \
        black_box_variational_inference(model.lnprob_theta_i, D,x0 , time_series, t, num_samples=200)
        init_mean    = -1 * np.ones(D)
        init_log_std = -5 * np.ones(D)
        init_var_params = np.concatenate([init_mean, init_log_std])
        variational_params = adam(gradient, init_var_params, step_size=0.1, num_iters=500)
        total_thetas.append(variational_params.tolist())
   

print (time_series)
print (total_thetas)
print (weights)
weights = np.array(weights).reshape((len(time_series),num_particles))

ess = []
for i in range(len(time_series)):
    w_i = 0
    tmp = weights[i]
    tmp = tmp/sum(tmp)
    for w in tmp:
        w_i += w**2
    ess.append(1./w_i)
    
import matplotlib.pyplot as plt
means = np.array(total_thetas)[:,0]
sds = np.sqrt(np.exp(np.array(total_thetas)[:,1]))

plt.plot(range(len(time_series)),time_series,color='orange')
plt.plot(range(len(time_series)),means,color='blue')
plt.fill_between(range(len(time_series)),means+ 2*sds,means- 2*sds,alpha=.3)
plt.show()


    


        
    
    
    
    
    
    
    
    
