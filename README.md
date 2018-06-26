![](https://raw.githubusercontent.com/hsamuelson/rnetf/master/rnetffLogo.JPG)


## Description
Training of neural networks using backpropagation, resilient backpropagation with (Riedmiller, 1994). Using neuralnet package. The package allows flexible settings through custom-choice of error and activation function. Furthermore, the calculation of generalized weights (Intrator O & Intrator N, 1993) is implemented. This package provides a framework to run neural networks and other classification algorithms through a sub-classification regression in order to perfect, and tune accuracy both recall, in the given model.

## Problem
Training sets with a large number of individual objects, which lowers neural net performance.

## Solution
This problem is solved by declaring columns sub classes / classes separately of the final class. For example if we needed to identify an individual car, we could simply say identify the the brand. After identifying the brand say it is a BMW we can simply remove all rows from the training set which contain individual car id's which do not belong to a BMW. From there we then can retrain a new neural net algorithm to identify the model type of the BMW, which was previously an impossible task due to the number of options. After identifying the model of BMW, we can now train a new neural net in order to identify the individual car id, this too was previously impossible due to the number of rows with a unique id. 

## Versions
### rnetf01Compiled.R
Easy plug and play for any algorithm though it is designed for a neural network. Because this is plug and play you will have to choose your own classification algorithm.
### RNetf-Lib01.R
This has a built in full neural network, optimized to work with the generic rnet framework. This is a work in progress and is still in dev.

## Installation 
Rnetf01Compiled.R, Rnetf-Lib01.R and standalonefunctions.R can all be run from R studio by downloading the git but they can also be run using the ```source("")``` command, on a RAW link. 
#### Examples
```
#Call Rnetf-Lib01.R
source("https://raw.githubusercontent.com/hsamuelson/rnetf/master/RNetf-Lib01.R")

#Call rnetf01Compiled.R
source("https://raw.githubusercontent.com/hsamuelson/rnetf/master/rnetf01Compiled.R")
```
## Usage 
```{r}
mexNet(hiddenMode = 3, final.id = 1, classColumn.range = 2:4, Dat = idenity, inputDat = 0, thres = 0.01 )
```


## Arguments

### hiddenMode
choose 1,2,3 and to decide method for choosing # of hidden nodes

     1     # of hidden nodes = # of inputnodes 
     2     # of hidden nodes = # of (# of inputnodes + output nodes)/ (2/3)
     3     # of hidden nodes =  sqrt(# of input nodes * # of atributes)
                
### final.id
The final column / class  you want to identify. Should be in Dat[x,] fourm, input x

### classColumn.range
The columns positions in the training set, which are other larger classes then final.id. Input should be in "a:b" fourm

### Dat
Training data, should be in the fourm of a table with, indenities as individual rows and attributes as columns. preferably in a data.frame().

### InputDat
The data you want to classify, should be a single array, with matching column ids, with the exclusion of the column to be trained for.
#### thres
The initail threshold for individual created neuralnets. Standard value (0.01)


