# Banking Data #####
## Data Set-Up ####

setwd("C:/Users/joosl/Downloads/Courses/Operations Analytics/")

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
library(neuralnet)
library(nnet)
library(e1071)


bank <- read_csv("bank_small.csv")

barplot(table(bank$y),
        main = "Bar Plot of Term Deposit Subscription",
        xlab = "Subscribed",
        ylab = "Frequency",
        col = c("red", "blue"))

table(bank$y)

# Load attrition data
df <- bank %>% mutate_if(is.ordered, factor, ordered = FALSE)
head(df)


# Create training (80%) and test (20%) sets
set.seed(123)  # for reproducibility
bank_split <- initial_split(df, prop = 0.8, strata = "y")
#If we want to explicitly control the sampling so that our training and test 
#sets have similar y distributions, we can use stratified sampling
bank_train <- training(bank_split)
bank_test  <- testing(bank_split)


## Logistic Regression #####
names(getModelInfo())

#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
bank_log <- train(
  y ~ ., 
  data = bank_train,
  method = "glm", 
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)


# Print results
print(bank_log$results)
bank_log
summary(bank_log)


bank_test$pred <- predict(bank_log, bank_test)
table(predicted=bank_test$pred, actual=bank_test$y) #classification table / confusion matrix (contingency table

conf_matrix <- table(predicted=bank_test$pred, actual=bank_test$y) #classification table / confusion matrix (contingency table


## Model with only significant variables ####
#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
bank_log2 <- train(
  y ~ contact + day + month + duration + loan + campaign + job + poutcome, 
  data = bank_train,
  method = "glm", 
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)


# Print results
print(bank_log2$results)
bank_log2
summary(bank_log2)




## create dummies for significant features #####
df2 <- bank %>% mutate_if(is.ordered, factor, ordered = FALSE)

# contact
unique(df2$contact)
df2$contact_unknown <- ifelse(df2$contact == "unknown", 1, 0)
df2$contact_cellular <- ifelse(df2$contact == "cellular", 1, 0)

df2$contact_telephone <- ifelse(df2$contact == "telephone", 1, 0)

# month
unique(df2$month)
df2$month_feb <- ifelse(df2$month == "feb", 1, 0)
df2$month_jan <- ifelse(df2$month == "jan", 1, 0)
df2$month_jul <- ifelse(df2$month == "jul", 1, 0)
df2$month_jun <- ifelse(df2$month == "jun", 1, 0)
df2$month_mar <- ifelse(df2$month == "mar", 1, 0)
df2$month_oct <- ifelse(df2$month == "oct", 1, 0)
df2$month_sep <- ifelse(df2$month == "sep", 1, 0)

df2$month_may <- ifelse(df2$month == "may", 1, 0)
df2$month_aug <- ifelse(df2$month == "aug", 1, 0)
df2$month_apr <- ifelse(df2$month == "apr", 1, 0)
df2$month_nov <- ifelse(df2$month == "nov", 1, 0)
df2$month_dec <- ifelse(df2$month == "dec", 1, 0)

# job
unique(df2$job)
df2$job_retired <- ifelse(df2$job == "retired", 1, 0)
df2$job_admin <- ifelse(df2$job == "admin.", 1, 0)

df2$job_unemployment <- ifelse(df2$job == "unemployed", 1, 0)
df2$job_services <- ifelse(df2$job == "services", 1, 0)
df2$job_management <- ifelse(df2$job == "management", 1, 0)
df2$job_bluecollar <- ifelse(df2$job == "blue-collar", 1, 0)
df2$job_selfemployed <- ifelse(df2$job == "self-employed", 1, 0)
df2$job_technician <- ifelse(df2$job == "technician", 1, 0)
df2$job_entrepreneur <- ifelse(df2$job == "entrepreneur", 1, 0)
df2$job_student <- ifelse(df2$job == "student", 1, 0)
df2$job_housemaid <- ifelse(df2$job == "housemaid", 1, 0)
df2$job_unknown <- ifelse(df2$job == "unknown", 1, 0)

# poutcome
unique(df2$poutcome)
df2$poutcome_other <- ifelse(df2$poutcome == "other", 1, 0)
df2$poutcome_success <- ifelse(df2$poutcome == "success", 1, 0)
df2$poutcome_failure <- ifelse(df2$poutcome == "failure", 1, 0)

df2$poutcome_unknown <- ifelse(df2$poutcome == "unknown", 1, 0)


set.seed(123)  # for reproducibility
bank_split2 <- initial_split(df2, prop = 0.8, strata = "y")
bank_train2 <- training(bank_split2)
bank_test2  <- testing(bank_split2)


## Model with only significant variables (dummies) ####
#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
bank_log3 <- train(
  y ~ 
    contact_unknown + contact_cellular + 
    day + 
    month_feb + month_jan + month_jul + month_jun + month_mar + month_oct + month_sep +
    duration + 
    loan + 
    campaign + 
    job_retired + job_admin +
    poutcome_other + poutcome_success + poutcome_failure, 
  data = bank_train2,
  method = "glm", 
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)


# Print results
print(bank_log3$results)
bank_log3
summary(bank_log3)


## Further Tuning ####
#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
bank_log4 <- train(
  y ~ 
    contact_unknown + 
    day + 
    month_feb + month_jan + month_jul + month_jun + month_mar + month_oct + month_sep +
    duration + 
    loan + 
    job_retired + 
    poutcome_other + poutcome_success, 
  data = bank_train2,
  method = "glm", 
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)


# Print results
print(bank_log4$results)
bank_log4
summary(bank_log4)



## Model with only significant variables (dummies) - including all dummies per variable ####
#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
bank_log5 <- train(
  y ~ 
    contact_unknown + contact_cellular + 
    day + 
    month_feb + month_jul + month_jun + month_mar + month_oct # + month_jan
    + month_sep + month_may + month_aug + month_apr + month_nov + month_dec +
    duration + 
    loan + 
    campaign + 
    job_retired + job_admin + job_services #+ job_unemployment 
     + job_management + 
    job_bluecollar + job_selfemployed + job_technician + job_entrepreneur + 
    job_student + job_housemaid + job_unknown + 
    poutcome_other #+ poutcome_success 
  + poutcome_failure + poutcome_unknown, 
  data = bank_train2,
  method = "glm", 
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)


# Print results
print(bank_log5$results)
bank_log5
summary(bank_log5)


## Further Tuning ####
#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
bank_log6 <- train(
  y ~ 
    contact_unknown  + 
    day + 
    month_feb + month_jun + month_mar + month_oct # + month_jan
    + month_sep + month_may + month_aug + month_apr + month_dec +
    duration + 
    loan + 
    campaign + 
    job_retired #+ job_unemployment 
    + poutcome_other #+ poutcome_success 
  + poutcome_failure + poutcome_unknown, 
  data = bank_train2,
  method = "glm", 
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)


# Print results
print(bank_log6$results)
bank_log6
summary(bank_log6)

print(bank_log4$coefnames)
print(bank_log6$coefnames)

print(bank_log4$results)
print(bank_log6$results)
print(bank_log$results)

bank_test2$pred <- predict(bank_log6, bank_test2)
table(predicted=bank_test2$pred, actual=bank_test2$y) #classification table / confusion matrix (contingency table)

conf_matrix6 <- table(predicted=bank_test2$pred, actual=bank_test2$y) #classification table / confusion matrix (contingency table)



## Error Metrics 1st and 6th Model #######
print(bank_log)
bank_log

### Model 1 ####
# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
print(paste("Accuracy:", accuracy))

# Calculate precision
precision <- conf_matrix[2, 2] / sum(conf_matrix[, 2])
print(paste("Precision:", precision))

# Calculate recall (sensitivity)
recall <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
print(paste("Recall:", recall))

# Calculate F1-score
f1_score <- 2 * precision * recall / (precision + recall)
print(paste("F1-score:", f1_score))

# Balanced Accuracy
sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])
balanced_accuracy <- (sensitivity + specificity) / 2
print(paste("Balanced Accuracy:", balanced_accuracy))

### Model 6 ####
# Calculate accuracy
accuracy6 <- sum(diag(conf_matrix6)) / sum(conf_matrix6)
print(paste("Accuracy 6:", accuracy6))

# Calculate precision
precision6 <- conf_matrix6[2, 2] / sum(conf_matrix6[, 2])
print(paste("Precision 6:", precision6))

# Calculate recall (sensitivity)
recall6 <- conf_matrix6[2, 2] / sum(conf_matrix6[2, ])
print(paste("Recall 6:", recall6))

# Calculate F1-score
f1_score6 <- 2 * precision6 * recall6 / (precision6 + recall6)
print(paste("F1-score 6:", f1_score6))

# Balanced Accuracy
sensitivity6 <- conf_matrix6[2, 2] / sum(conf_matrix6[2, ])
specificity6 <- conf_matrix6[1, 1] / sum(conf_matrix6[1, ])
balanced_accuracy6 <- (sensitivity6 + specificity6) / 2
print(paste("Balanced Accuracy 6:", balanced_accuracy6))







## SVM ####
#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
bank_svm <- train(
  y ~ ., 
  data = bank_train,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# Print results
print(bank_svm$results)
bank_svm

#Plotting the results, we see that smaller values of the cost parameter
#( C≈ 2–8) provide better cross-validated accuracy scores for these 
#training data:
ggplot(bank_svm) + theme_light()

bank_test$svm_pred = predict(bank_svm, bank_test)
table(predicted=bank_test$svm_pred, actual=bank_test$y) #classification table / confusion matrix (contingency table)

conf_matrix_s1 <- table(predicted=bank_test$svm_pred, actual=bank_test$y) #classification table / confusion matrix (contingency table)



## SVM with SIGNIFICANT values ####
#caret’s train() function with method = "svmRadialSigma" is used to get 
#values of C (cost) and \sigma (related with the \gamma of Radial Basis function)
#through cross-validation
set.seed(1854)  # for reproducibility
bank_svm2 <- train(
  y ~ 
    contact_unknown  + 
    day + 
    month_feb + month_jun + month_mar + month_oct # + month_jan
  + month_sep + month_may + month_aug + month_apr + month_dec +
    duration + 
    loan + 
    campaign + 
    job_retired #+ job_unemployment 
  + poutcome_other #+ poutcome_success 
  + poutcome_failure + poutcome_unknown, 
  data = bank_train2,
  method = "svmRadial",               
  preProcess = c("center", "scale"),  #x's standardized (i.e.,centered around zero with a sd of one)
  trControl = trainControl(method = "cv", number = 10),
  tuneLength = 10
)

# Print results
print(bank_svm2$results)
bank_svm2
ggplot(bank_svm2) + theme_light()

bank_test2$svm_pred = predict(bank_svm2, bank_test2)
table(predicted=bank_test2$svm_pred, actual=bank_test2$y) #classification table / confusion matrix (contingency table)

conf_matrix_s2 <- table(predicted=bank_test2$svm_pred, actual=bank_test2$y) #classification table / confusion matrix (contingency table)



## Error Metrics SVM Models #######
### SVM Model 1 ####
# Calculate accuracy
accuracy_s1 <- sum(diag(conf_matrix_s1)) / sum(conf_matrix_s1)
print(paste("Accuracy SVM 1:", accuracy_s1))

# Calculate precision
precision_s1 <- conf_matrix_s1[2, 2] / sum(conf_matrix_s1[, 2])
print(paste("Precision SVM 1:", precision_s1))

# Calculate recall (sensitivity)
recall_s1 <- conf_matrix_s1[2, 2] / sum(conf_matrix_s1[2, ])
print(paste("Recall SVM 1:", recall_s1))

# Calculate F1-score
f1_score_s1 <- 2 * precision_s1 * recall_s1 / (precision_s1 + recall_s1)
print(paste("F1-score SVM 1:", f1_score_s1))

# Balanced Accuracy
sensitivity_s1 <- conf_matrix_s1[2, 2] / sum(conf_matrix_s1[2, ])
specificity_s1 <- conf_matrix_s1[1, 1] / sum(conf_matrix_s1[1, ])
balanced_accuracy_s1 <- (sensitivity_s1 + specificity_s1) / 2
print(paste("Balanced Accuracy SVM Model 1:", balanced_accuracy_s1))



### SVM Model 2 ####
# Calculate accuracy
accuracy_s2 <- sum(diag(conf_matrix_s2)) / sum(conf_matrix_s2)
print(paste("Accuracy SVM 2:", accuracy_s2))

# Calculate precision
precision_s2 <- conf_matrix_s2[2, 2] / sum(conf_matrix_s2[, 2])
print(paste("Precision SVM 2:", precision_s2))

# Calculate recall (sensitivity)
recall_s2 <- conf_matrix_s2[2, 2] / sum(conf_matrix_s2[2, ])
print(paste("Recall SVM 2:", recall_s2))

# Calculate F1-score
f1_score_s2 <- 2 * precision_s2 * recall_s2 / (precision_s2 + recall_s2)
print(paste("F1-score SVM 2:", f1_score_s2))

# Balanced Accuracy
sensitivity_s2 <- conf_matrix_s2[2, 2] / sum(conf_matrix_s2[2, ])
specificity_s2 <- conf_matrix_s2[1, 1] / sum(conf_matrix_s2[1, ])
balanced_accuracy_s2 <- (sensitivity_s2 + specificity_s2) / 2
print(paste("Balanced Accuracy SVM Model 2:", balanced_accuracy_s2))



## Neural Nets 1 ####
# Load the neuralnet package
library(neuralnet)

set.seed(1)

bank_train2$loan <- ifelse(bank_train2$loan == "yes", 1, 0)
bank_test2$loan <- ifelse(bank_test2$loan == "yes", 1, 0)


# Create a formula for the neural network model
# formula_nn <- as.formula("bikes_hired ~ .")
formula_nn <- as.formula("y ~ contact_unknown  + 
    day +  month_feb + month_jun + month_mar + month_oct + month_sep + 
    month_may + month_aug + month_apr + month_dec +
    duration + loan + campaign + 
    job_retired + poutcome_other + poutcome_failure + poutcome_unknown")


# Create the neural network model
nn_model <- neuralnet(formula_nn, data = bank_train2, hidden = 1)  # You can adjust the number of hidden layers and neurons as per your requirements

# Make predictions on the test set
nn_predictions <- predict(nn_model, bank_test2)

# Extract predicted values
bank_test2$prob_pred_nn <- nn_predictions[,2]

bank_test2$pred_nn <- 
  as.numeric(bank_test2$prob_pred_nn >= 0.5)

conf_matrix_nn <- table(predicted=bank_test2$pred_nn, actual=bank_test2$y) #classification table / confusion matrix (contingency table)
conf_matrix_nn

plot(nn_model)


### Accuracy ####
# Calculate accuracy
accuracy_nn <- sum(diag(conf_matrix_nn)) / sum(conf_matrix_nn)
print(paste("Accuracy NN 1:", accuracy_nn))

# Calculate precision
precision_nn <- conf_matrix_nn[2, 2] / sum(conf_matrix_nn[, 2])
print(paste("Precision NN 1:", precision_nn))

# Calculate recall (sensitivity)
recall_nn <- conf_matrix_nn[2, 2] / sum(conf_matrix_nn[2, ])
print(paste("Recall NN 1:", recall_nn))

# Calculate F1-score
f1_score_nn <- 2 * precision_nn * recall_nn / (precision_nn + recall_nn)
print(paste("F1-score NN:", f1_score_nn))

# Balanced Accuracy
sensitivity_nn <- conf_matrix_nn[2, 2] / sum(conf_matrix_nn[2, ])
specificity_nn <- conf_matrix_nn[1, 1] / sum(conf_matrix_nn[1, ])
balanced_accuracy_nn <- (sensitivity_nn + specificity_nn) / 2
print(paste("Balanced Accuracy NN 1:", balanced_accuracy_nn))





## Neural Nets 3 ####
set.seed(1854)
# Create a formula for the neural network model
# formula_nn <- as.formula("bikes_hired ~ .")
formula_nn2 <- as.formula("y ~ contact_unknown  + 
    day +  month_feb + month_jun + month_mar + month_oct + month_sep + 
    month_may + month_aug + month_apr + month_dec +
    duration + loan + campaign + 
    job_retired + poutcome_other + poutcome_failure + poutcome_unknown")


# Create the neural network model
nn_model2 <- neuralnet(formula_nn2, data = bank_train2, hidden = 3)  # You can adjust the number of hidden layers and neurons as per your requirements

# Make predictions on the test set
nn_predictions2 <- predict(nn_model2, bank_test2)

# Extract predicted values
bank_test2$prob_pred_nn2 <- nn_predictions2[,2]

bank_test2$pred_nn2 <- 
  as.numeric(bank_test2$prob_pred_nn2 >= 0.5)

conf_matrix_nn2 <- table(predicted=bank_test2$pred_nn2, actual=bank_test2$y) #classification table / confusion matrix (contingency table)
conf_matrix_nn2

# Calculate the total number of predictions
total_predictions2 <- sum(conf_matrix2)

# Calculate the number of correct predictions
correct_predictions2 <- sum(diag(conf_matrix2))

# Calculate accuracy
accuracy2 <- correct_predictions2 / total_predictions2

# Print accuracy
print(paste("Accuracy:", accuracy2))

plot(nn_model2)

### Accuracy ####
# Calculate accuracy
accuracy_nn2 <- sum(diag(conf_matrix_nn2)) / sum(conf_matrix_nn2)
print(paste("Accuracy NN 2:", accuracy_nn2))

# Calculate precision
precision_nn2 <- conf_matrix_nn2[2, 2] / sum(conf_matrix_nn2[, 2])
print(paste("Precision NN 2:", precision_nn2))

# Calculate recall (sensitivity)
recall_nn2 <- conf_matrix_nn2[2, 2] / sum(conf_matrix_nn2[2, ])
print(paste("Recall NN 2:", recall_nn2))

# Calculate F1-score
f1_score_nn2 <- 2 * precision_nn2 * recall_nn2 / (precision_nn2 + recall_nn2)
print(paste("F1-score NN 2:", f1_score_nn2))

# Balanced Accuracy
sensitivity_nn2 <- conf_matrix_nn2[2, 2] / sum(conf_matrix_nn2[2, ])
specificity_nn2 <- conf_matrix_nn2[1, 1] / sum(conf_matrix_nn2[1, ])
balanced_accuracy_nn2 <- (sensitivity_nn2 + specificity_nn2) / 2
print(paste("Balanced Accuracy NN 2:", balanced_accuracy_nn2))


## Neural Nets 5 ####
set.seed(1854)
# Create a formula for the neural network model
# formula_nn <- as.formula("bikes_hired ~ .")
formula_nn2 <- as.formula("y ~ contact_unknown  + 
    day +  month_feb + month_jun + month_mar + month_oct + month_sep + 
    month_may + month_aug + month_apr + month_dec +
    duration + loan + campaign + 
    job_retired + poutcome_other + poutcome_failure + poutcome_unknown")


# Create the neural network model
nn_model2 <- neuralnet(formula_nn2, data = bank_train2, hidden = 5)  # You can adjust the number of hidden layers and neurons as per your requirements

# Make predictions on the test set
nn_predictions2 <- predict(nn_model2, bank_test2)

# Extract predicted values
bank_test2$prob_pred_nn2 <- nn_predictions2[,2]

bank_test2$pred_nn2 <- 
  as.numeric(bank_test2$prob_pred_nn2 >= 0.5)

conf_matrix_nn2 <- table(predicted=bank_test2$pred_nn2, actual=bank_test2$y) #classification table / confusion matrix (contingency table)
conf_matrix_nn2

# Calculate the total number of predictions
total_predictions2 <- sum(conf_matrix2)

# Calculate the number of correct predictions
correct_predictions2 <- sum(diag(conf_matrix2))

# Calculate accuracy
accuracy2 <- correct_predictions2 / total_predictions2

# Print accuracy
print(paste("Accuracy:", accuracy2))

plot(nn_model2)

### Accuracy ####
# Calculate accuracy
accuracy_nn2 <- sum(diag(conf_matrix_nn2)) / sum(conf_matrix_nn2)
print(paste("Accuracy NN 2:", accuracy_nn2))

# Calculate precision
precision_nn2 <- conf_matrix_nn2[2, 2] / sum(conf_matrix_nn2[, 2])
print(paste("Precision NN 2:", precision_nn2))

# Calculate recall (sensitivity)
recall_nn2 <- conf_matrix_nn2[2, 2] / sum(conf_matrix_nn2[2, ])
print(paste("Recall NN 2:", recall_nn2))

# Calculate F1-score
f1_score_nn2 <- 2 * precision_nn2 * recall_nn2 / (precision_nn2 + recall_nn2)
print(paste("F1-score NN 2:", f1_score_nn2))

# Balanced Accuracy
sensitivity_nn2 <- conf_matrix_nn2[2, 2] / sum(conf_matrix_nn2[2, ])
specificity_nn2 <- conf_matrix_nn2[1, 1] / sum(conf_matrix_nn2[1, ])
balanced_accuracy_nn2 <- (sensitivity_nn2 + specificity_nn2) / 2
print(paste("Balanced Accuracy NN 2:", balanced_accuracy_nn2))
