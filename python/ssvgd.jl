
observation_variance = 1
transition_variance = 10



function lnprob(theta_i)
	lnprob_theta_i(self, theta_i, theta_t_minus_1, time_series,t,iter_):
        #ln poisson observations
            lnprob_theta_i = 1.0/(np.sqrt(2*np.pi*observation_variance))*np.exp(-.5*(1.0/observation_variance)*((time_series[t] - theta_i )**2))

            transition_sum = 0
	    for theta_t_minus_1_i in theta_t_minus_1:

                transition_sum += 1.0/(np.sqrt(2*np.pi*transition_variance))*np.exp(-.5*(1.0/transition_variance)*((theta_i - theta_t_minus_1_i )**2))
                
            return (np.log(lnprob_theta_i)+np.log(transition_sum))




time_series = [1,2,3,4]

