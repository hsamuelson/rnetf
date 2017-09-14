#
# Henry Samuelson 3/12/17
#
# Auto find and train neural net.
  #This is part of a larger grouping of projects (mexnet) to automate ann analysis

  #HiddenSelect 
    #Used by other functions, to determine number of hidden nodes

  #nCompute()
    #Will create neuralNet and then process test data. Can take muiltiple inputs
    #Auto finds and creates training formula.
    #Tipically used for one time use nn's.

  #nTrain()
    #Trains a neual net that can be used later and, works with  the 
    #standard neuralnet(), function compute()
    #has no input test data processing ability.

#' A ml modular framework
#'
#' This function allows you to express your love of cats.
#' @name rnetf rnetf
#' @param rnetf type stuff
#' @keywords neural net
#' @export
#' @examples
#' nCompute()
#' nTrain()
if (!require("neuralnet")) install.packages("neuralnet")
library(neuralnet)
hiddenSelect <- function(hidd, tDat){ 
  if(hidd == 1){
    hiddenMode <- round(length(colnames(tDat)) -1)
    
  } else if(hidd == 2){
    hiddenMode <- round((round(length(colnames(tDat)) -1) + 1)/(2/3))
  } else if(hidd == 3){
    hiddenMode <- round(sqrt(length(colnames(tDat)) -1) * length(tDat[,1]) )
  }
  return(hiddenMode)
}

nCompute <- function(trainDat = 0, hidd = 0, final.id = 0, thres = 0.01, testDat = 0){
  if(sum(trainDat) == 0 | sum(hidd) == 0 | sum(final.id) == 0 | sum(testDat) == 0){
    return("Missing Inputs!")
  } 
  library(neuralnet)
  final.id <- colnames(trainDat[final.id])
  #For colname need a new variable
  half2 <- trainDat
  half2[final.id] <- NULL
  model_formula <- paste(paste(final.id), '~', paste(colnames(half2), collapse = "+", sep = "_"))
  return(compute(neuralnet(data = trainDat, hidden = hiddenSelect(hidd, trainDat), threshold = thres, formula = model_formula), testDat)$net.result)
}


nTrain <- function(trainDat = 0, hidd = 0, final.id = 0, thres = 0.01){
  if(sum(trainDat) == 0 | sum(hidd) == 0 | sum(final.id) == 0){
    return("Missing Inputs!")
  } 
  library(neuralnet)
  final.id <- colnames(trainDat[final.id])
  #For colname need a new variable
  half2 <- trainDat
  half2[final.id] <- NULL
  model_formula <- paste(paste(final.id), '~', paste(colnames(half2), collapse = "+", sep = "_"))
  return(neuralnet(data = trainDat, hidden = hiddenSelect(hidd, trainDat), threshold = thres, formula = model_formula))
}



####WORK IN PROGRESS ####
nComputeAdd <- function(trainDat = 0, hidd = 0, final.id = 0, thres = 0.01, testDat = 0){
  if(sum(trainDat) == 0 | sum(hidd) == 0 | sum(final.id) == 0 | sum(testDat) == 0){
    return("Missing Inputs!")
  } 
  library(neuralnet)
  final.id <- colnames(trainDat[final.id])
  #For colname need a new variable
  half2 <- trainDat
  half2[final.id] <- NULL
  model_formula <- paste(paste(final.id), '~', paste(colnames(half2), collapse = "+", sep = "_"))
  reslutz <- compute(neuralnet(data = trainDat, hidden = hiddenSelect(hidd, trainDat), threshold = thres, formula = model_formula), testDat)$net.result
  
  
  return(reslutz)
}
