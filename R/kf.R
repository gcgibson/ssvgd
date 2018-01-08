library(dlm)

data <- seq(1,10) + rnorm(10,0,1)

mod <- dlmModPoly(1,dV = .1,dW=10)
filt<-dlmFilter(data,mod)

v <- dropFirst(unlist(dlmSvd2var(filt$U.C, filt$D.C)))

n_part <- 10

Sys.setenv(PATH = paste("/Users/gcgibson/anaconda/bin", Sys.getenv("PATH"), sep=":"))
exec_str <- 'python /Users/gcgibson/Stein-Variational-Gradient-Descent/python/locally_level_gaussian.py '
#exec_str <- 'python python/locally_level_gaussian.py '
exec_str <- paste(exec_str, toString(data))
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


aggregate_forecast <- matrix(first_state_forecasts,nrow=length(data),ncol=n_part,byrow = TRUE)


meanSsvgdForecasts <-c()
lowPiSsvgdForecasts <- c()
highPiSsvgdForecasts <- c()

for (i in 1:nrow(aggregate_forecast)){
  meanSsvgdForecasts <- c(meanSsvgdForecasts,mean(aggregate_forecast[i,]))
  srted <- sort(aggregate_forecast[i,])
  lowPiSsvgdForecasts <- c(lowPiSsvgdForecasts,srted[round(.05*length(data))+1])
  highPiSsvgdForecasts <- c(highPiSsvgdForecasts,srted[round(.95*length(data))-1])
}

filter_results <- c(dropFirst(filt$f),11)

library(ggplot2)

p1<- ggplot() 
p1<- p1+ geom_line(data = data.frame(x=seq(1,length(filter_results)),y=filter_results), aes(x = x, y = y), color = "red") +
  
  geom_line(data=data.frame(x=seq(1,length(data)),y=data), aes(x = x, y = y), color = "cornflowerblue") +
  geom_ribbon(data=data.frame(x=seq(1,length(data)),y=filter_results),aes(x=x,ymin=filter_results-1.96*sqrt(v),ymax=filter_results+1.96*sqrt(v)),alpha=0.3)+
  xlab('data_date') +
  ylab('count') #+ ylim(low=-2,high=10) 
print(p1)


p<- ggplot() 
p<- p+ geom_line(data = data.frame(x=seq(1,length(meanSsvgdForecasts)),y=meanSsvgdForecasts), aes(x = x, y = y), color = "red") +
  
  geom_line(data=data.frame(x=seq(1,length(data)),y=data), aes(x = x, y = y), color = "cornflowerblue") +
  geom_ribbon(data=data.frame(x=seq(1,length(data)),y=meanSsvgdForecasts),aes(x=x,ymin=lowPiSsvgdForecasts,ymax=highPiSsvgdForecasts,alpha=0.3))+
  xlab('data_date') +
  ylab('count')# +ylim(low=-10,high=100) 
print(p)

library(cowplot)
plot_grid(p, p1, labels = c("SVGD","PF"))
