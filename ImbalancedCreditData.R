library(DMwR) #For smote
library(ROSE) #For rose
library(caret)

card.data <- read.csv("creditcard.csv",header=FALSE,sep=",",na.strings = '?')
card.data <- apply(card.data, 2, function(x) as.numeric(x))
card.data <- as.data.frame(card.data)
head(card.data,3)
card.data <- na.omit(card.data)
head(card.data,3)
names(card.data)[length(card.data)] <- "Class"
card.data$Class <- as.factor(card.data$Class)
levels(card.data$Class) <- c("Ok", "Fraud")
table(card.data$Class)

inTrainRows <- createDataPartition(card.data$Class,p=0.8,list=FALSE)
trainData <- card.data[inTrainRows,]
testData <-  card.data[-inTrainRows,]
table(trainData$Class)
table(testData$Class)
down_train <- downSample(x = trainData[, -ncol(trainData)],
                         y = trainData$Class)
table(down_train$Class)
up_train <- upSample(x = trainData[, -ncol(trainData)],
                     y = trainData$Class)
table(up_train$Class)

smote_train <- SMOTE(Class ~ ., data  = trainData)                         
table(smote_train$Class) 

rose_train <- ROSE(Class ~ ., data  = trainData)$data                         
table(rose_train$Class) 

ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

orig_fit <- train(Class ~ ., data = trainData, 
                  method = "rpart",
                  metric = "ROC",
                  trControl = ctrl)

down_fit <- train(Class ~ ., data = down_train, 
                      method = "rpart",
                      metric = "ROC",
                      trControl = ctrl)

up_fit <- train(Class ~ ., data = up_train, 
                    method = "rpart",
                    metric = "ROC",
                    trControl = ctrl)

rose_fit <- train(Class ~ ., data = rose_train, 
                      method = "rpart",
                      metric = "ROC",
                      trControl = ctrl)

smote_fit <- train(Class ~ ., data = smote_train, 
                       method = "rpart",
                       metric = "ROC",
                       trControl = ctrl)

models <- list(original = orig_fit,
                       down = down_fit,
                       up = up_fit,
                       SMOTE = smote_fit,
                       ROSE = rose_fit)

resampling <- resamples(models)

test_roc <- function(model, data) {
  library(pROC)
  roc_obj <- roc(data$Class, 
                 predict(model, data, type = "prob")[, "Ok"],
                 levels = c("Fraud", "Ok"))
  ci(roc_obj)
}

ppp<-predict(down_fit, newdata=testData[,-31],type="raw")
table(ppp,testData)

test <- lapply(models, test_roc, data = testData)
test <- lapply(test, as.vector)
test <- do.call("rbind", test)
colnames(test) <- c("lower", "ROC", "upper")
test <- as.data.frame(test)

summary(resampling, metric = "ROC")
test
