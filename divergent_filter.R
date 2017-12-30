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


aggregate_forecast <- matrix(first_state_forecasts,nrow=5,ncol=10,byrow = TRUE)


meanSsvgdForecasts <-c()
lowPiSsvgdForecasts <- c()
highPiSsvgdForecasts <- c()

for (i in 1:nrow(aggregate_forecast)){
  meanSsvgdForecasts <- c(meanSsvgdForecasts,mean(aggregate_forecast[i,]))
  srted <- sort(aggregate_forecast[i,])
  lowPiSsvgdForecasts <- c(lowPiSsvgdForecasts,srted[2])
  highPiSsvgdForecasts <- c(highPiSsvgdForecasts,srted[9])
}

lowPf <-summ_smc$x$f$quant$`0.025`
highPf <- summ_smc$x$f$quant$`0.975`



p1<- ggplot() 
p1<- p1+ geom_line(data = data.frame(x=seq(1,length(summ_smc$x$f$mean)),y=exp(summ_smc$x$f$mean)), aes(x = x, y = y), color = "red") +
  
  geom_line(data=data.frame(x=seq(1,length(ma_data)),y=ma_data), aes(x = x, y = y), color = "cornflowerblue") +
  geom_ribbon(data=data.frame(x=seq(1,length(ma_data)),y=exp(summ_smc$x$f$mean)),aes(x=x,ymin=exp(lowPf),ymax=exp(highPf)),alpha=0.3)+
  xlab('data_date') +
  ylab('count') + ylim(low=-10,high=10) 
print(p1)




p<- ggplot() 
p<- p+ geom_line(data = data.frame(x=seq(1,length(meanSsvgdForecasts)),y=exp(meanSsvgdForecasts)), aes(x = x, y = y), color = "red") +
  
  geom_line(data=data.frame(x=seq(1,length(ma_data)),y=ma_data), aes(x = x, y = y), color = "cornflowerblue") +
  geom_ribbon(data=data.frame(x=seq(1,length(ma_data)),y=exp(meanSsvgdForecasts)),aes(x=x,ymin=exp(lowPiSsvgdForecasts),ymax=exp(highPiSsvgdForecasts),alpha=0.3))+
  xlab('data_date') +
  ylab('count') +ylim(low=-10,high=10) 
print(p)

library(cowplot)
plot_grid(p, p1, labels = c("SVGD","PF"))

