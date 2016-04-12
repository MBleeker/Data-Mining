rm(list=ls(all=TRUE))
library(forecast)
setwd("/home/liam/cloud/uni/dm/assign1/Xys")
Xfiles <- dir(getwd(), pattern ="X.csv")
yfiles <- dir(getwd(), pattern ="y.csv")
testSubsetfiles <- dir(getwd(), pattern ="Subset.csv")
X <- list()
y <- list()
testSubset <- list()
preds <- list()
rmse_all = 0
for(i in 1:length(Xfiles)){
  X[[i]] <- read.csv(Xfiles[i],header=TRUE)
  y[[i]] <- read.csv(yfiles[i],header=TRUE)
  testSubset[[i]] <- read.csv(testSubsetfiles[i],header=FALSE)
  colnames(y[[i]]) = 'mood'
  colnames(X[[i]])[1] = 'mood_lag1'
  x <- ts(X[[i]])
  mood <- ts(y[[i]])
  subset = min(testSubset[[i]])+1
  fit = auto.arima(mood[1:subset,], xreg = x[1:subset,1])
  if(any(is.na(fit[["coef"]][2]))){
    fit = auto.arima(mood[1:subset,])
    preds[[i]] = predict(fit, nrow(x) - subset)$pred
  } else {
    preds[[i]] = predict(fit, newxreg = x[(subset+1):nrow(x),1])$pred
  }
  err = preds[[i]] - mood[(subset+1):nrow(mood),1]
  rmse = sqrt(mean(err^2))
  rmse_all = rmse_all + rmse
  write.csv(preds[[i]], paste('/home/liam/cloud/uni/dm/assign1/arima_preds/arima_preds_',
                              gsub('_X.csv','',Xfiles[i]),'.csv',sep=""))
}
rmse_all/length(X)