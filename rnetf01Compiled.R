mexNet <- function (
  hiddenMode = 3,
  final.id = 1,
  classColumn.range = 2:4,
  Dat = idenity,
  inputDat = 0,
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
    net.results <- compute(nn, inputDat) #equationPLus[1,]
    net.results <- round(net.results$net.result)
    
    #Alternative to -which()
    holder <- numeric(0) 
    for(j in 1:length(Dat[,1])){
      if(Dat[j, classColumn.range.names[lowest[1,1]]] == net.results){
        holder <- rbind.data.frame(holder, Dat[j,])
      }
    }
    Dat <- holder
    
  }
  #train for final.id
  model_formula <- paste(paste(colnames(Dat[final.id])), '~', paste(col_names, collapse = '+', sep = ''))
  nn <- neuralnet(model_formula, Dat, threshold = thres, hidden = hiddenSelect(hiddenMode) )
  net.results <- compute(nn, inputDat) #equationPLus[1,]
  net.results <- round(net.results$net.result)
  #net.results 
  return(net.results)
}