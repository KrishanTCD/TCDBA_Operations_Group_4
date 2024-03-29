rm(list=ls())
#graphics.off()
# Helper packages
library(dplyr)     # for data wrangling
library(ggplot2)   # for awesome graphics
library(rsample)   # for data splittingg
library(modeldata) #package that includes couple of useful datasets

# Modeling packages
library(caret)    # for classification and regression training
library(kernlab)  # for fitting SVMs
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

split.index <- floor(nrow(clean_bike)*0.8)
train.df <- clean_bike[1:split.index, ]
valid.df <- clean_bike[(split.index + 1):nrow(clean_bike), ]


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


#insignificant: week, cloud cover, snow depth 
#remove them from the model 

model2 <- lm(bikes_hired ~ year + wday + month + humidity + pressure +  precipitation + 
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


#### model three, excluding weather variables: 

model3 <- lm(bikes_hired ~ humidity + pressure +  precipitation + 
               sunshine + mean_temp + min_temp, 
             data = train.df)

summary(model3)

#predictions test set: 
predictions31 <- predict(model3, train.df)
#error metrics
RMSE(predictions31, train.df$bikes_hired)
R2(predictions31, train.df$bikes_hired)
MAPE(predictions31, train.df$bikes_hired)
MAE(predictions31, train.df$bikes_hired)
MSE(predictions31, train.df$bikes_hired)



#prediction on validation 
predictions32 <- predict(model3, valid.df)
# Model performance
RMSE(predictions32, valid.df$bikes_hired)
R2(predictions32, valid.df$bikes_hired)
MAPE(predictions32, valid.df$bikes_hired)
MAE(predictions32, valid.df$bikes_hired)
MSE(predictions32, valid.df$bikes_hired)


####neural networks: 
library(neuralnet)



#normalise values: 

library(caret)
norm.values <- preProcess(train.df, method="range")
train.norm.df <- predict(norm.values, train.df)
valid.norm.df <- predict(norm.values, valid.df)

## drop the categorical variables from the data: 
train.norm.df <- train.norm.df[, -c(3,4)]
valid.norm.df <- valid.norm.df[, -c(3,4)]

### create neural network with all variablles: 
nn <- neuralnet(bikes_hired ~ ., 
                data = train.norm.df, linear.output = T, 
                hidden = c(4,4))
plot(nn)

#training
training.prediction <- compute(nn,train.norm.df)

R2(training.prediction$net.result, train.norm.df$bikes_hired)
RMSE(training.prediction$net.result, train.norm.df$bikes_hired)
MAPE(training.prediction$net.result, train.norm.df$bikes_hired)
MAE(training.prediction$net.result, train.norm.df$bikes_hired)
MSE(training.prediction$net.result, train.norm.df$bikes_hired)


#validation 
validation.prediction <- compute(nn,valid.norm.df)

R2(validation.prediction$net.result,valid.norm.df$bikes_hired)
RMSE(validation.prediction$net.result,valid.norm.df$bikes_hired)
MAPE(validation.prediction$net.result, valid.norm.df$bikes_hired)
MAE(validation.prediction$net.result, valid.norm.df$bikes_hired)
MSE(validation.prediction$net.result, valid.norm.df$bikes_hired)


#new neural network with better error metrics, drop same values as those insignificant from linear: 
nn1 <- neuralnet(bikes_hired ~ humidity + pressure + radiation + precipitation + 
                   sunshine + mean_temp + min_temp, 
                  data = train.norm.df, 
                  hidden = c(4,4))
plot(nn1)

#training
training.prediction1 <- compute(nn1,train.norm.df)

R2(training.prediction1$net.result,train.norm.df$bikes_hired)
RMSE(training.prediction1$net.result,train.norm.df$bikes_hired)
MAPE(training.prediction1$net.result,train.norm.df$bikes_hired)
MAE(training.prediction1$net.result,train.norm.df$bikes_hired)
MSE(training.prediction1$net.result,train.norm.df$bikes_hired)


#validation 
validation.prediction1 <- compute(nn1,valid.norm.df)

R2(validation.prediction1$net.result,valid.norm.df$bikes_hired)
RMSE(validation.prediction1$net.result,valid.norm.df$bikes_hired)
MAPE(validation.prediction1$net.result, valid.norm.df$bikes_hired)
MAE(validation.prediction1$net.result, valid.norm.df$bikes_hired)
MSE(validation.prediction1$net.result, valid.norm.df$bikes_hired)

#assess distribution day: 
avg_day <- tapply(clean_bike$bikes_hired, clean_bike$wday, mean)
barplot(avg_day, 
        main = "Average Rentals by day of the week", 
        xlab = "Day of the week", 
        ylab = "Average hired",
        col = "skyblue")
       

#assess distribution: 
avg_month <- tapply(clean_bike$bikes_hired, clean_bike$month, mean)
barplot(avg_month, 
        main = "Average Rentals by month", 
        xlab = "Month", 
        ylab = "Average hired",
        col = "skyblue")


#play with neural network sizes: 
### create neural network with all variablles: 
nn3 <- neuralnet(bikes_hired ~ ., 
                data = train.norm.df, linear.output = T, 
                hidden = c(3,3))
plot(nn3)

#training

validation.prediction3 <- compute(nn3,valid.norm.df)

R2(validation.prediction3$net.result,valid.norm.df$bikes_hired)
RMSE(validation.prediction3$net.result,valid.norm.df$bikes_hired)
MAPE(validation.prediction3$net.result,valid.norm.df$bikes_hired)






