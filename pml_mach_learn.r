setwd('/home/khalid/Documents/03 - Courses/Practical Machine Learning/barbell project/data');
training_raw = read.csv('pml-training.csv')
testing_raw = read.csv('pml-testing.csv')
# head(training_raw)
# summary(training_raw)
# str(training_raw)

# NB: As the model is going to be tested against the test-set let us consider only those attributes which are
# present in the test set and which show some variation (therefore can hold classification value) in the
# test set. This is also a useful step to reduce the number of attributes.
summary(testing_raw)

####### PRE-PROCESSING #############

# Convert all variables test set to factors
col_names <- names(testing_raw)
testing_as_factors <- lapply(testing_raw[,col_names], factor)
# Get the number of factors for each column
num_factor_levels <- lapply(testing_as_factors, function(x) nlevels(x))
# Keep only those column which have >1 factor level (so some classification value)
training_proc <- training_raw[, num_factor_levels > 1]

####### DATA-SPLITTING #############

# Create a test set (10%), a training set (70%)
# and a cross validation set (20%). Test set not to be used
# for any of training procedures

library(caret)
set.seed(42)
inTraining <- createDataPartition(training_proc$classe, p = 0.9, list=FALSE)
barbell_train <- training_proc[inTraining,]
barbell_test <- training_proc[-inTraining,]

set.seed(42)
inValidation <- createDataPartition(barbell_train$classe, p=(0.2/0.9), list=FALSE)
barbell_valid <- barbell_train[inValidation,]
barbell_train <- barbell_train[-inValidation,]

####### TRAIN-MODEL #############

# Train model on the data using all of the remaining factors on the test set
# Using a random forest model (because in the lectures this is supposed to be reasonable)
set.seed(42)
modFit <- train(classe ~ ., method='rf', data=barbell_train)

####### CHECK ACCURACY ##########

# Initially check on training set
# Output standard finalModel output which will give confusion matrix as well as other performance metrics
modFit$finalModel
# This shows just a single miss-classification error

# Checking the training data through the prediction again suggests there is no (0) classification error
train_prediction <- predict(modFit, newdata=barbell_train)
confusionMatrix(data=train_prediction, barbell_train$classe)

# Then check on validation set
validation_prediction <- predict(modFit, newdata=subset(barbell_valid, select=-c(classe)))
confusionMatrix(data=validation_prediction, barbell_valid$classe)
# Which again gives 0 miss-classification error

# Finally; as results above are so positive; try with the test set
final_prediction <- predict(modFit, newdata=subset(barbell_test, select=-c(classe)))
confusionMatrix(data=final_prediction, barbell_test$classe)

# Here there was a single miss-classification error, but with an accuracy of 99.95% it would be
# unlucky for the model to fail at fitting one of the 20 examples


####### MAKE PREDICTIONS ##########

# Extract relevant fields of real test data:
testing_proc <- testing_raw[, num_factor_levels > 1]
real_test_prediction <- predict(modFit, newdata=testing_proc)
real_test_prediction