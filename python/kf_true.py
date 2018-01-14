from pydlm import dlm, trend, seasonality
from scipy.stats import norm
import numpy as np
# A linear trend
linear_trend = trend(degree=1, discount=1, name='linear_trend', w=100)
# A seasonality
time_series = [1,0,4,3]
simple_dlm = dlm(time_series) + linear_trend

simple_dlm.fit()
filteredMean = simple_dlm.getMean(filterType='forwardFilter')
filteredVar = simple_dlm.getVar(filterType='forwardFilter')

ll = 0
for i in range(len(time_series)):
       ll += np.log(norm.pdf(time_series[i],filteredMean[i],filteredVar[i]))

print (ll)
