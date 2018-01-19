from pydlm import dlm, trend, seasonality
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# A linear trend
linear_trend = trend(degree=1, discount=1, name='linear_trend', w=10)
# A seasonality

time_series = []
for i in range(10):
    if i == 0:
        x_sim = np.random.normal(0,1,1)
    else:
        x_sim = np.random.normal(x_sim,10,1)
    time_series.append(np.random.normal(x_sim,10,1))
time_series = np.array(time_series)
simple_dlm = dlm(time_series) + linear_trend

simple_dlm.fit()
filteredMean = simple_dlm.getMean(filterType='forwardFilter')
filteredVar = simple_dlm.getVar(filterType='forwardFilter')

ll = 0
one_step_ahead_samples = []
for i in range(len(time_series)):
    tmp_samples = []
    for j in range(1000):
        tmp = np.random.normal(filteredMean[i],filteredVar[i], 1)
        tmp_samples.append(np.random.normal(tmp,1,1))
    one_step_ahead_samples.append(tmp_samples)
one_step_ahead_samples = np.array(one_step_ahead_samples)

upper_pi = []
lower_pi = []
for p in one_step_ahead_samples:
    upper_pi.append(np.percentile(p,95))
    lower_pi.append(np.percentile(p,5))

time_series_shifted = time_series
#plt.plot(range(len(time_series_shifted)),time_series_shifted,color='orange')
#plt.fill_between(range(len(time_series_shifted)),upper_pi,lower_pi,alpha=.3)
#plt.show()



from pykalman import KalmanFilter
random_state = np.random.RandomState(0)

transition_matrix = 1
transition_offset = .1
observation_matrix = 1
observation_offset = 1
transition_covariance = 10
observation_covariance = 1
initial_state_mean = 0
initial_state_covariance = 1

# sample from model
kf = KalmanFilter(
                  transition_matrix, observation_matrix, transition_covariance,
                  observation_covariance, transition_offset, observation_offset,
                  initial_state_mean, initial_state_covariance,
                  random_state=random_state
                  )
filtered_state_means, filtered_state_variances = kf.filter(time_series)


filteredMean = filtered_state_means.reshape((-1))

filteredVar = filtered_state_variances.reshape((-1))




one_step_ahead_samples = []
for i in range(len(time_series)):
    tmp_samples = []
    for j in range(10000):
        tmp = np.random.normal(filteredMean[i],filteredVar[i], 1)
        tmp2 = np.random.normal(tmp,10,1)
        
        tmp_samples.append(np.random.normal(tmp2,10,1))
    one_step_ahead_samples.append(tmp_samples)
one_step_ahead_samples = np.array(one_step_ahead_samples)

upper_pi = []
lower_pi = []
for p in one_step_ahead_samples:
    upper_pi.append(np.percentile(p,95))
    lower_pi.append(np.percentile(p,5))

time_series = time_series.reshape((-1))
time_series_shifted = time_series.tolist()[1:] + [10]

plt.plot(range(len(time_series_shifted)),time_series_shifted,color='orange')
plt.fill_between(range(len(time_series_shifted)),upper_pi,lower_pi,alpha=.3)
plt.show()











