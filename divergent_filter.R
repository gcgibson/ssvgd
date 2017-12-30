require(rbiips)
library(MCMCpack)
locally_level_model_file = '/Users/gcgibson/Stein-Variational-Gradient-Descent/locally_level_1.bug' # BUGS model filename

ma_data = c(1,4,0,3,2)
t_max = length(ma_data)


#setting the mean value of the initial count to 1400
locally_level_data = list(t_max=t_max, y = ma_data,  mean_x_init=-10)
locally_level_model = biips_model(locally_level_model_file, data=locally_level_data,sample_data = FALSE)
n_part = 10000 # Number of particles
variables = c('x') # Variables to be monitored
mn_type = 'fs'; rs_type = 'stratified'; rs_thres = 0.5 # Optional parameters


out_smc = biips_smc_samples(locally_level_model, variables, n_part,
                            type=mn_type, rs_type=rs_type, rs_thres=rs_thres)
diag_smc = biips_diagnosis(out_smc)

summ_smc = biips_summary(out_smc, probs=c(.025, .975))
print (exp(summ_smc$x$f$mean))

Sys.setenv(PATH = paste("/Users/gcgibson/anaconda/bin", Sys.getenv("PATH"), sep=":"))

exec_str <- 'python /Users/gcgibson/Stein-Variational-Gradient-Descent/python/locally_level.py '
exec_str <- paste(exec_str, toString(ma_data))
print (exec_str)
ssvgdForecasts <- system(exec_str,intern=TRUE,wait = TRUE)

#ssvgdForecasts <- strsplit(ssvgdForecasts,",")
#ssvgdForecasts <- as.numeric(unlist(ssvgdForecasts))
#ssvgdForecasts
count <- 1
ssvgdForecasts <- strsplit(ssvgdForecasts[1],",")
ssvgdForecasts <- unlist(ssvgdForecasts)
first_state_forecasts <- c()

for (i in seq(1,length(ssvgdForecasts)) ){
  first_state_forecasts <- c(first_state_forecasts,as.numeric(ssvgdForecasts[i]))
}


meanSsvgdForecasts <- exp(first_state_forecasts)
lowPiSsvgdForecasts <- c()
highPiSsvgdForecasts <- c()

for (i in 1:nrow(aggregate_forecast)){
  meanSsvgdForecasts <- c(meanSsvgdForecasts,mean(aggregate_forecast[i,]))
  srted <- sort(aggregate_forecast[i,])
  lowPiSsvgdForecasts <- c(lowPiSsvgdForecasts,srted[2])
  highPiSsvgdForecasts <- c(highPiSsvgdForecasts,srted[9])
}
