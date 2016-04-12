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
  xreg_vars = c('call_lag1', 'sms_lag1')
  xregs = x[1:subset,xreg_vars]
  # uncomment below line to fit without xregs
  #xregs = NULL
  # fit an auto arima
  fit = auto.arima(mood[1:subset,], xreg = xregs)
  if(any(is.na(fit[["coef"]][2])) | is.null(xregs)){
    xregs <- NULL
    fit = auto.arima(mood[1:subset,])
  } else {
    xregs = x[(subset+1):nrow(x),xreg_vars]
  }
  # predict first value
  preds[[i]] = predict(fit, nrow(x) - subset, newxreg = xregs)$pred
  preds[[i]][2:length(preds[[i]])] = 0
  for (j in 1:(length(preds[[i]])-1)){
    # fit another arima, force parameters, then predict one step ahead from this
    if (!is.null(xregs)){
      refit = Arima(mood[1:(subset+j),], order = fit$arma[c(1, 6, 2)], 
                    fixed = fit$coef, xreg = x[1:(subset+j),xreg_vars])
      preds[[i]][j+1] = predict(refit, 1, newxreg = t(x[subset+j+1,xreg_vars]))$pred
    } else {
      refit = Arima(mood[1:(subset+j),], order = fit$arma[c(1, 6, 2)], fixed = fit$coef)
      preds[[i]][j+1] = predict(refit, 1)$pred
    }
  }
  print(refit$coef)
  err = preds[[i]] - mood[(subset+1):nrow(mood),1]
  rmse = sqrt(mean(err^2))
  rmse_all = rmse_all + rmse
  write.csv(preds[[i]], paste('/home/liam/cloud/uni/dm/assign1/arima_preds/arima2_preds_',
                              gsub('_X.csv','',Xfiles[i]),'.csv',sep=""))
}
rmse_all/length(X)