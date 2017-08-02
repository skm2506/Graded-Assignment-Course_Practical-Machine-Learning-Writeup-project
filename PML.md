---
title: "Graded Assignment:Course_Practical Machine Learning Writeup project"
author: "Shashikesh Mishra"
date: "1 August 2017"
output: html_document
---


# Index
* An Introduction
* Local machine directory and Enviroment setup
* Data loading and Cleaning
* Exploratory Data Analysis
* Data Visualisation
* Correlation Analysis
* Model building
* Conclusion

# Synopsis
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants.

### An Intro
The goal of our project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. We may use any of the other variables to predict with. We should create a report describing how you built your model, how we used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

### Data Source
The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

The Training data set for this project is available here.
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data set for this project is available here.
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# Local machine directory and Enviroment setup


```r
rm(list = ls())
setwd("C:/Users/sk_mi/Desktop/onlinelearning/Coursera/practicalmachinelearning")
getwd()
```

```
## [1] "C:/Users/sk_mi/Desktop/onlinelearning/Coursera/practicalmachinelearning"
```


```r
set.seed(12345)
library(caret)
library(knitr)
library(rpart)
library(rpart.plot)
library(rattle)
library(randomForest)
library(corrplot)
library(repmis)
```

# Loading and Cleaning of data

```r
# Uploading data from respective site
urlTrain<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
```

Now as we have downloaded data set from the site in to our enviroment, now let's creat read the data set and then we will creat partion of training data set in to two subset(60% & 30%) for our model building process. Test data set will remain untouched for our final result.


```r
# By using read.csv function we will read and view the downloaded data set.
training<- read.csv(url(urlTrain))
testing<- read.csv(url(urlTest))
```


```r
# Let's creat a prtition in training data set.
indata<- createDataPartition(training$classe, p = 0.7, list = FALSE)
Train_data_set<- training[indata, ]
Test_data_set<- training[-indata, ]
dim(Train_data_set)
```

```
## [1] 13737   160
```

```r
dim(Test_data_set)
```

```
## [1] 5885  160
```

### Now we will remove near zero or NA values from both the data set.

```r
remove_NA<- nearZeroVar(Train_data_set)
Train_data_set<- Train_data_set[, -remove_NA]
Test_data_set<- Test_data_set[, -remove_NA]
dim(Train_data_set)
```

```
## [1] 13737   106
```

```r
dim(Test_data_set)
```

```
## [1] 5885  106
```


```r
# Let's remove variable which mostly contain NA
NA_var<- sapply(Train_data_set, function(x) mean(is.na(x)))>0.95
Train_data_set<- Train_data_set[, NA_var==FALSE]; Test_data_set<- Test_data_set[, NA_var==FALSE]
dim(Train_data_set); dim(Test_data_set)
```

```
## [1] 13737    59
```

```
## [1] 5885   59
```

```r
#Just one more cleaning we are done.
Train_data_set<- Train_data_set[, -(1:5)]; Test_data_set<- Test_data_set[, -(1:5)]  
dim(Train_data_set); dim(Test_data_set) # by removing identification variables
```

```
## [1] 13737    54
```

```
## [1] 5885   54
```
Afterall removing all unnecessary variable from the data set we are left with 54 odd variable.

## Exploratory data analysis
under this we will explore the data to find correlation b/w variables. We will also visualize data set to understand the pattern of data set.

```r
corMatrix <- cor(Train_data_set[, -54])
corrplot(corMatrix, order = "hclust", method = "ellipse", type = "lower", tl.cex = 0.6, tl.col = rgb(0, 0, 0))
```

![plot of chunk unnamed-chunk-9](Graded-Assignment-Course_Practical-Machine-Learning-Writeup-project/figures/Rplot-11-1.png)

Under this correlation plot, It's clearing visible that which variables are mostly correlated to each other. Thus moving further let's start building model in next step.
??
## Model Building to accomplish prediction.
As we do remember that our goal is to creat a model in the manner that best suited to predict 20 different test cases.In order to accomplish this task we are going to use prediction medel such as RandomForest, Decision Tree and GB MOdel.

### A. Decision Tree


```r
control <- trainControl(method = "cv", number = 3)
fit_rpart <- train(classe ~ ., data = Train_data_set, method = "rpart", 
                   trControl = control)
print(fit_rpart, digits = 4)
```

```
## CART 
## 
## 13737 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 9158, 9159, 9157 
## Resampling results across tuning parameters:
## 
##   cp       Accuracy  Kappa  
##   0.03794  0.5627    0.43653
##   0.05539  0.4173    0.20770
##   0.11586  0.3374    0.08131
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03794.
```


```r
fancyRpartPlot(fit_rpart$finalModel)
```

![plot of chunk unnamed-chunk-11](figures/unnamed-chunk-11-1.png)



```r
# Now we will predict outcomes using validation set
predict_rpart <- predict(fit_rpart, Test_data_set)
# Show prediction result
(conf_rpart <- confusionMatrix(Test_data_set$classe, predict_rpart))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1494   21  128    0   31
##          B  470  380  289    0    0
##          C  467   29  530    0    0
##          D  416  184  324    0   40
##          E   95   90  219    0  678
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5237          
##                  95% CI : (0.5109, 0.5365)
##     No Information Rate : 0.4999          
##     P-Value [Acc > NIR] : 0.0001376       
##                                           
##                   Kappa : 0.3791          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5078  0.53977  0.35570       NA   0.9052
## Specificity            0.9388  0.85350  0.88714   0.8362   0.9213
## Pos Pred Value         0.8925  0.33363  0.51657       NA   0.6266
## Neg Pred Value         0.6561  0.93173  0.80243       NA   0.9852
## Prevalence             0.4999  0.11963  0.25319   0.0000   0.1273
## Detection Rate         0.2539  0.06457  0.09006   0.0000   0.1152
## Detection Prevalence   0.2845  0.19354  0.17434   0.1638   0.1839
## Balanced Accuracy      0.7233  0.69664  0.62142       NA   0.9133
```


```r
(accuracy_rpart <- conf_rpart$overall[1])
```

```
##  Accuracy 
## 0.5237043
```

this is clearly visible that accuracy rate is 0.53 which is 50% around. so decision tree model is not suitalble for this predicton.
Now next we will start with Randon forest.

### B. Random Forest

```r
set.seed(12345)
fitmod.rf<- train(classe ~., data = Train_data_set, method="rf", trControl=control)
print(fitmod.rf, digits = 4)
```

```
## Random Forest 
## 
## 13737 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## Summary of sample sizes: 9159, 9158, 9157 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa 
##    2    0.9924    0.9903
##   27    0.9945    0.9931
##   53    0.9920    0.9899
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```


```r
# Now we will predict the outcomes using validation set
predict_rf <- predict(fitmod.rf, Test_data_set)

# Show prediction result
(conf_rf <- confusionMatrix(Test_data_set$classe, predict_rf))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    5 1133    1    0    0
##          C    0    4 1022    0    0
##          D    0    0    8  956    0
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9946, 0.9978)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9955          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9970   0.9965   0.9913   0.9969   1.0000
## Specificity            1.0000   0.9987   0.9992   0.9984   0.9994
## Pos Pred Value         1.0000   0.9947   0.9961   0.9917   0.9972
## Neg Pred Value         0.9988   0.9992   0.9981   0.9994   1.0000
## Prevalence             0.2853   0.1932   0.1752   0.1630   0.1833
## Detection Rate         0.2845   0.1925   0.1737   0.1624   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9985   0.9976   0.9952   0.9976   0.9997
```

Now lets have a look on the accuracy of Random Forest model on overall data set.


```r
(accuracy_rf <- conf_rf$overall[1])
```

```
##  Accuracy 
## 0.9964316
```


```r
# Let's plot the Random Forest model on outcome variables.
plot(conf_rf$table, col = conf_rf$byClass, 
     main = paste("Random Forest - Accuracy =",
                  round(conf_rf$overall['Accuracy'], 4)))
```

![plot of chunk unnamed-chunk-17](figures/unnamed-chunk-17-1.png)

### C. Generalized Boosted Medel

```r
# Let's predict the train data set by using this final model
set.seed(12345)
controlgbm <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
fitmod.gbm  <- train(classe ~ ., data=Train_data_set, method = "gbm",trControl = controlgbm, verbose=FALSE)
fitmod.gbm$finalModel
```

```
## A gradient boosted model with multinomial loss function.
## 150 iterations were performed.
## There were 53 predictors of which 43 had non-zero influence.
```


```r
# Let's predict the outcome on test data set
predict_gbm<- predict(fitmod.gbm, Test_data_set)
(conf_gbm<- confusionMatrix(Test_data_set$classe, predict_gbm))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670    2    0    1    1
##          B    9 1117   11    2    0
##          C    0   20 1004    2    0
##          D    3    2   14  944    1
##          E    0    1    3   11 1067
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9859          
##                  95% CI : (0.9825, 0.9888)
##     No Information Rate : 0.2858          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9822          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9929   0.9781   0.9729   0.9833   0.9981
## Specificity            0.9990   0.9954   0.9955   0.9959   0.9969
## Pos Pred Value         0.9976   0.9807   0.9786   0.9793   0.9861
## Neg Pred Value         0.9972   0.9947   0.9942   0.9967   0.9996
## Prevalence             0.2858   0.1941   0.1754   0.1631   0.1816
## Detection Rate         0.2838   0.1898   0.1706   0.1604   0.1813
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9960   0.9867   0.9842   0.9896   0.9975
```


```r
# Let's plot the GBM on outcome variables.
plot(conf_gbm$table, col = conf_gbm$byClass, 
     main = paste("GBM - Accuracy =", round(conf_gbm$overall['Accuracy'], 4)))
```

![plot of chunk unnamed-chunk-20](figures/unnamed-chunk-20-1.png)

# Conclusion
as we can see the accuracy rate for Random Forest is 0.9965 which is more accurate than decision tree & GBM. This may be because the variable are much more correlated with each other in model. Thus, it leads us to our highest accuracy rate with final result. 

The accuracy result of all three regression models are listed below;
Random Forest : 0.9963
Decision Tree : 0.7368
GBM : 0.9839


```r
# Noe Let's apply best suited model to test data set.
(predict(fitmod.rf, testing))
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```












