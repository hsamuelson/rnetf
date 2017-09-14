#' @name rnetf
#' rnetf
#' 
#' Description
#' 
#' Training of neural networks using backpropagation, resilient backpropagation with (Riedmiller, 1994). Using neuralnet package. The package allows flexible settings through custom-choice of error and activation function. Furthermore, the calculation of generalized weights (Intrator O & Intrator N, 1993) is implemented. This package provides a framework to run neural networks and other classification algorithms through a sub-classification regression in order to perfect, and tune accuracy both recall, in the given model.
#' 
#' Problem
#' 
#' Training sets with a large number of individual objects, which lowers neural net performance.
#' 
#' Solution
#' 
#' This problem is solved by declaring columns sub classes / classes separately of the final class. For example if we needed to identify an individual car, we could simply say identify the the brand. After identifying the brand say it is a BMW we can simply remove all rows from the training set which contain individual car id's which do not belong to a BMW. From there we then can retrain a new neural net algorithm to identify the model type of the BMW, which was previously an impossible task due to the number of options. After identifying the model of BMW, we can now train a new neural net in order to identify the individual car id, this too was previously impossible due to the number of rows with a unique id.
#' 
#' Versions
#' 
#' rnetf01Compiled.R
#' 
#' Easy plug and play for any algorithm though it is designed for a neural network. Because this is plug and play you will have to choose your own classification algorithm.
#' 
#' RNetf-Lib01.R
#' 
#' This has a built in full neural network, optimized to work with the generic rnet framework. This is a work in progress and is still in dev.
#' 
#' Installation
#' 
#' Rnetf01Compiled.R, Rnetf-Lib01.R and standalonefunctions.R can all be run from R studio by downloading the git but they can also be run using the source("") command, on a RAW link.
#' 
#' Examples
#' 
#' #Call Rnetf-Lib01.R
#' source("https://raw.githubusercontent.com/hsamuelson/rnetf/master/RNetf-Lib01.R")
#' 
#' #Call rnetf01Compiled.R
#' source("https://raw.githubusercontent.com/hsamuelson/rnetf/master/rnetf01Compiled.R")
#' Usage
#' 
#' mexNet(hiddenMode = 3, final.id = 1, classColumn.range = 2:4, Dat = idenity, inputDat = 0, thres = 0.01 )
#' @param 
#' 
#' hiddenMode
#' 
#' choose 1,2,3 and to decide method for choosing # of hidden nodes
#' 
#' 1     # of hidden nodes = # of inputnodes 
#' 2     # of hidden nodes = # of (# of inputnodes + output nodes)/ (2/3)
#' 3     # of hidden nodes =  sqrt(# of input nodes * # of atributes)
#' final.id
#' 
#' The final column / class you want to identify. Should be in Dat[x,] fourm, input x
#' 
#' classColumn.range
#' 
#' The columns positions in the training set, which are other larger classes then final.id. Input should be in "a:b" fourm
#' 
#' Dat
#' 
#' Training data, should be in the fourm of a table with, indenities as individual rows and attributes as columns. preferably in a data.frame().
#' 
#' InputDat
#' 
#' The data you want to classify, should be a single array, with matching column ids, with the exclusion of the column to be trained for.
#' 
#' thres
#' 
#' The initail threshold for individual created neuralnets. Standard value (0.01)
#' @author hsamuelson
#' @examples 
#' rnetf()
#' 

if (!require("neuralnet")) install.packages("neuralnet")
library(neuralnet)
rnetf<- function (
hiddenMode = 3,
final.id = 1,
classColumn.range = 2:4,
Dat = idenity,
inputDat = 0,              #has to NOT have both class idenityies, in order to predict them. 
thres = 0.01 ) {
  
  if(sum(Dat) == 0){
    return("ERROR No Dat: Dat == 0")
  }
  if(final.id == 0){
    return("ERROR: Final.id == 0: No Column to train for selected")
  }
  if(sum(classColumn.range) == 0){
    return("ERROR: no classcolumn.range")
  }
  if(sum(inputDat) == 0){
    return("No input Data")
  }
  #Setting up classrange
  equationPLus <- Dat #what we will take columnnames() from to generate C~A+B
  if(sum(classColumn.range) == 0 ){
    #Then all columns will be considered for subclasses
    print("All columns will be considered for SubClasses")
    
    classColumn.range <- Dat
    classColumn.range[,final.id] <- NULL
    return("This function is not supported yet.... Please provide a value for classColumn.range")
    
  } else{
    holder <- equationPLus
    
    for( i in classColumn.range){
      holder[paste(colnames(equationPLus[i]))] <- NULL

    }
    equationPLus <- holder
    
    classColumn.range <- Dat[,classColumn.range]
  }
  #Create final.id
  equationPLus[,final.id] <- NULL
  
  
  hiddenSelect <- function(hidd){ #w will be Dat
    if(hidd == 1){
      hiddenMode <- round(length(colnames(equationPLus)))
      
    } else if(hidd == 2){
      hiddenMode <- round((round(length(colnames(equationPLus))) + 1)/(2/3))
    } else if(hidd == 3){
      hiddenMode <- round(sqrt(length(colnames(equationPLus))*  length(Dat[,1]) ))
    }
    return(hiddenMode)
  }
  
  classColumn.range.names <- colnames(classColumn.range)
  classColumn.range.values <- numeric(0)
  for(i in 1:length(classColumn.range[1,])){
    classColumn.range.values[i] <- length(table(classColumn.range[,i]))  
  }
  
  order.Set <- rbind.data.frame((1:length(classColumn.range.names)), classColumn.range.values) #hash map for names the valules in the 1:length() will corilate to classColumn.names
  order.Set <- order.Set[order(order.Set[2,])]
  
  setValuefor.Order.Set <- length(order.Set[1,]) #Needs to be counted here before it is cutdown
  for(i in 1: setValuefor.Order.Set){ #will loop for all the subclasses
    
    if(length(order.Set) == 0){
      print("breaking order.Set == 0")
      break
    }
    
    
    lowest <- order.Set[1]
    order.Set[1] <- NULL
    
    col_names <- colnames(equationPLus)
    #Train algorithum
    model_formula <- paste(paste(classColumn.range.names[lowest[1,1]]), '~', paste(col_names, collapse = '+', sep = ''))
    nn <- neuralnet(model_formula, Dat, threshold = thres, hidden = hiddenSelect(hiddenMode) )
    #plot(nn) #just to visulize and test remove this for final
    net.results <- compute(nn, equationPLus[1,]) #inputDat) #equationPLus[1,]
    net.results <- round(net.results$net.result)
    
    #Alternative to -which()
    holder <- numeric(0) 
    for(j in 1:length(Dat[,1])){
      if(Dat[j, classColumn.range.names[lowest[1,1]]] == net.results){
        #print(j) #print when equal
        holder <- rbind.data.frame(holder, Dat[j,])
      }
    }
    Dat <- holder
   
  }
  #train for final.id
  model_formula <- paste(paste(colnames(Dat[final.id])), '~', paste(col_names, collapse = '+', sep = ''))
  nn <- neuralnet(model_formula, Dat, threshold = thres, hidden = hiddenSelect(hiddenMode) )
  net.results <- compute(nn, equationPLus[1,])#inputDat) #equationPLus[1,]
  net.results <- round(net.results$net.result)
  #net.results 
  return(net.results)
}
