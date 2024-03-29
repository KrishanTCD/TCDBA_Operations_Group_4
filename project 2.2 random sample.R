rm(list=ls())
#graphics.off()
# Helper packages
library(dplyr)     # for data wrangling
library(ggplot2)   # for awesome graphics
library(rsample)   # for data splittingg
library(modeldata) #package that includes couple of useful datasets

# Modeling packages
library(caret)    # for classification and regression training
library(kernlab)  # for fitting SVM
library(readr)

library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
install.packages('neuralnet')
library(neuralnet)
library(nnet)
library(e1071)
library("MLmetrics")
install.packages("MLmetrics")
install.packages("rsq")
library(rsq)


bike <- read.csv("/Users/sarahraubenheimer/Downloads/bikes_hires.csv")
clean_bike <- na.omit(bike)
#clean_bike <- clean_bike[, -1]
#split the data chronologically 
set.seed(100)
# order data chronologically 
clean_bike <- clean_bike[order(clean_bike$date), ]
clean_bike <- clean_bike[, -1]

#random sample
set.seed(100)
train.index<-sample(c(1:dim(clean_bike)[1]), dim(clean_bike)[1]*0.8)  
valid.index<-setdiff(c(1:dim(clean_bike)[1]),clean_bike)
train.df<-clean_bike[train.index,]
valid.df<-clean_bike[valid.index,]



#Linear regression
model <- lm(bikes_hired ~ ., 
            data = train.df)
summary(model)
model

predictions <-predict(model, valid.df)


# Model performance
RMSE(predictions, valid.df$bikes_hired)
R2(predictions, valid.df$bikes_hired)
MAPE(predictions, valid.df$bikes_hired)
MAE(predictions, valid.df$bikes_hired)
MSE(predictions, valid.df$bikes_hired)

#predictions training 
predictions1 <- predict(model, train.df)

RMSE(predictions1, train.df$bikes_hired)
R2(predictions1, train.df$bikes_hired)
MAPE(predictions1, train.df$bikes_hired)
MAE(predictions1, train.df$bikes_hired)
MSE(predictions1, train.df$bikes_hired)


#insignificant: week, cloud cover, snow depth + max temp 
#remove them from the model 

model2 <- lm(bikes_hired ~ year + wday + month + humidity + pressure +  precipitation + radiation +
               sunshine + mean_temp + min_temp, 
             data = train.df)

summary(model2)

#predictions test set: 
predictions21 <- predict(model2, train.df)
#error metrics
RMSE(predictions21, train.df$bikes_hired)
R2(predictions21, train.df$bikes_hired)
MAPE(predictions21, train.df$bikes_hired)
MAE(predictions21, train.df$bikes_hired)
MSE(predictions21, train.df$bikes_hired)



#prediction on validation 
predictions2 <- predict(model2, valid.df)
# Model performance
RMSE(predictions2, valid.df$bikes_hired)
R2(predictions2, valid.df$bikes_hired)
MAPE(predictions2, valid.df$bikes_hired)
MAE(predictions2, valid.df$bikes_hired)
MSE(predictions2, valid.df$bikes_hired)


