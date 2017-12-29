require(rbiips)
library(MCMCpack)
library(ggplot2)

ma_data <- round((sin(seq(10))**2)*10+10)

seasonal_model_file = '/Users/gcgibson/Desktop/lyme/seasonal_pois.bug' # BUGS model filename

seasonal_t_max = length(ma_data)
n_burn = 5000 # nb of burn-in/adaptation iterations
n_iter = 10000 # nb of iterations after burn-in
thin = 5 # thinning of MCMC outputs
n_part = 50 # nb of particles for the SMC
latent_names = c('x') # names of the variables updated with SMC and that need to be monitored

inits = list(-2)

seasonality =6
G = matrix(c(cos(2*pi/seasonality),sin(2*pi/seasonality),-sin(2*pi/seasonality),cos(2*pi/seasonality)), nrow=2, byrow=TRUE)

#setting the mean value of the initial count to 1400
seasonal_data = list(t_max=seasonal_t_max, y = ma_data,  G = G, mean_sigma_init = c(0,0), cov_sigma_init=.001*diag(2) ,mean_x_init=c(log(1400),log(1400)))
seasonal_model = biips_model(seasonal_model_file, data=seasonal_data,sample_data = FALSE)

##fixing variance for now, will extend model to handle inference over variance later

n_part = 100000 # Number of particles
variables = c('x1','x2','x3','x4') # Variables to be monitored
seasonal_out_smc = biips_smc_samples(seasonal_model, variables, n_part)
seasonal_model_summary = biips_summary(seasonal_out_smc, probs=c(.025, .975))

Sys.setenv(PATH = paste("/Users/gcgibson/anaconda/bin", Sys.getenv("PATH"), sep=":"))

exec_str <- 'python /Users/gcgibson/Stein-Variational-Gradient-Descent/python/time_series.py'
ssvgdForecasts <- system(exec_str,intern=TRUE,wait = TRUE)

#ssvgdForecasts <- strsplit(ssvgdForecasts,",")
#ssvgdForecasts <- as.numeric(unlist(ssvgdForecasts))
#ssvgdForecasts
count <- 1
ssvgdForecasts <- strsplit(ssvgdForecasts[1],",")
ssvgdForecasts <- unlist(ssvgdForecasts)
first_state_forecasts <- c()

for (i in seq(1,length(ssvgdForecasts),2) ){
  first_state_forecasts <- c(first_state_forecasts,as.numeric(ssvgdForecasts[i]))
}

aggregate_forecast <- matrix(first_state_forecasts,nrow=10,ncol=10,byrow = TRUE)

meanSsvgdForecasts <- c()
lowPiSsvgdForecasts <- c()
highPiSsvgdForecasts <- c()

for (i in 1:nrow(aggregate_forecast)){
  meanSsvgdForecasts <- c(meanSsvgdForecasts,mean(aggregate_forecast[i,]))
  srted <- sort(aggregate_forecast[i,])
  lowPiSsvgdForecasts <- c(lowPiSsvgdForecasts,srted[2])
  highPiSsvgdForecasts <- c(highPiSsvgdForecasts,srted[9])
}

lowPf <- pmax(0,seasonal_model_summary$x1$f$quant$`0.025`)
highPf <- pmin(ln(100),seasonal_model_summary$x1$f$quant$`0.975`)


p1<- ggplot() 
p1<- p1+ geom_line(data = data.frame(x=seq(1,length(seasonal_model_summary$x1$f$mean)),y=exp(seasonal_model_summary$x1$f$mean)), aes(x = x, y = y), color = "red") +
  
  geom_line(data=data.frame(x=seq(1,length(ma_data)),y=ma_data), aes(x = x, y = y), color = "cornflowerblue") +
  geom_ribbon(data=data.frame(x=seq(1,length(ma_data)),y=exp(seasonal_model_summary$x1$f$mean)),aes(x=x,ymin=exp(lowPf),ymax=exp(highPf)),alpha=0.3)+
  xlab('data_date') +
  ylab('count') + ylim(low=-10,high=110) 
print(p1)




p<- ggplot() 
p<- p+ geom_line(data = data.frame(x=seq(1,length(meanSsvgdForecasts)),y=exp(meanSsvgdForecasts)), aes(x = x, y = y), color = "red") +
  
  geom_line(data=data.frame(x=seq(1,length(ma_data)),y=ma_data), aes(x = x, y = y), color = "cornflowerblue") +
  geom_ribbon(data=data.frame(x=seq(1,length(ma_data)),y=exp(meanSsvgdForecasts)),aes(x=x,ymin=exp(lowPiSsvgdForecasts),ymax=exp(highPiSsvgdForecasts),alpha=0.3))+
  xlab('data_date') +
  ylab('count') +ylim(low=-10,high=110) 
print(p)

library(cowplot)
plot_grid(p, p1, labels = c("SVGD","PF"))


