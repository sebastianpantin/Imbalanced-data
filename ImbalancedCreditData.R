library(DMwR) #For smote
library(ROSE) #For rose
library(caret)
library(doMC)
registerDoMC(cores = 4)

card.data <- read.csv("creditcard.csv",header=FALSE,sep=",",na.strings = '?')
card.data <- apply(card.data, 2, function(x) as.numeric(x))
card.data <- as.data.frame(card.data)
card.data <- na.omit(card.data)
head(card.data,3)
names(card.data)[length(card.data)] <- "Class"
card.data$Class <- as.factor(card.data$Class)
levels(card.data$Class) <- c("Ok", "Fraud")
table(card.data$Class)

start_time <- Sys.time()
percentage = 0.2
B = 5
kappas <- matrix(0,B,5)
aucs <- matrix(0,B,5)

for (i in 1:B) {
inTrainRows <- createDataPartition(card.data$Class,p=percentage,list=FALSE)
trainData <- card.data[inTrainRows,]
testData <-  card.data[-inTrainRows,]
testDataY <- card.data[-inTrainRows,31]
table(trainData$Class)
table(testData$Class)
down_train <- downSample(x = trainData[, -ncol(trainData)],
                         y = trainData$Class)
#table(down_train$Class)
up_train <- upSample(x = trainData[, -ncol(trainData)],
                     y = trainData$Class)
#table(up_train$Class)

smote_train <- SMOTE(Class ~ ., data  = trainData)                         
#table(smote_train$Class) 

rose_train <- ROSE(Class ~ ., data  = trainData)$data                         
#table(rose_train$Class)

ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)

orig_fit <- train(Class ~ ., data = trainData, 
                  method = "rpart",
                  metric = "ROC",
                  tuneLength = 15,
                  trControl = ctrl)
print("OG fit")

down_fit <- train(Class ~ ., data = down_train, 
                      method = "rpart",
                      metric = "ROC",
                      trControl = ctrl)
print("down fit")

up_fit <- train(Class ~ ., data = up_train, 
                    method = "rpart",
                    metric = "ROC",
                    tuneLength = 15,
                    trControl = ctrl)
print("Up fit")

rose_fit <- train(Class ~ ., data = rose_train, 
                      method = "rpart",
                      metric = "ROC",
                      tuneLength = 15,
                      trControl = ctrl)
print("Rose fit")

smote_fit <- train(Class ~ ., data = smote_train, 
                       method = "rpart",
                       metric = "ROC",
                       tuneLength = 15,
                       trControl = ctrl)

print("smote fit")

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
}

predicts <- function(model,data) {
  pred_obj <- predict(model, data, type = "raw")
}

confus <- function(pred, obs) {
  conf_obj <- confusionMatrix(pred, obs)
}

kappa <- function(pred, obs) {
  kappa_obj <- confusionMatrix(pred, obs)$overall[2]
}
test <- lapply(models, test_roc, data = testData)
preds <- lapply(models, predicts, data = testData)
#confusions <- lapply(preds,confus,obs=testDataY)
kappa <- lapply(preds,kappa,obs=testDataY)
kappas[i,1] <- kappa$original
kappas[i,2] <- kappa$down
kappas[i,3] <- kappa$up
kappas[i,4] <- kappa$SMOTE
kappas[i,5] <- kappa$ROSE
aucs[i,1] <- test$original$auc[1]
aucs[i,2] <- test$down$auc[1]
aucs[i,3] <- test$up$auc[1]
aucs[i,4] <- test$SMOTE$auc[1]
aucs[i,5] <- test$ROSE$auc[1]
# plot(1:5,unlist(kappas), xlab="Sampling method", ylab = "Kappa", main = percentage, axes = FALSE)
# axis(1, 1:5, names(kappas))
# axis(2)
# box()
print(i)
}
boxplot(kappas,names=c("Original","Down", "Up","SMOTE", "ROSE"), ylab = "Kappa")
boxplot(aucs,names=c("Original","Down", "Up","SMOTE", "ROSE"), ylab = "ROC")
end_time <- Sys.time()
end_time - start_time

summary(resampling, metric = "ROC")

plot(test$original, col = 'red')
par(new=TRUE)
plot(test$down, col = 'orange')
par(new=TRUE)
plot(test$up, col = 'yellow')
par(new=TRUE)
plot(test$SMOTE, col = 'green')
par(new=TRUE)
plot(test$ROSE, col = 'blue')
legend(-0.1, 1, c("Original","Down","Up","Smote", "Rose"), col = c("red", "orange", "yellow", "green", "blue"),
       text.col = "green4", lty = 1,
       merge = TRUE)

confusions$original
confusions$up
confusions$down
confusions$ROSE
confusions$SMOTE
confusions$original$overall
confusions$up$overall
confusions$down$overall
confusions$ROSE$overall
confusions$SMOTE$overall
confusions$original$byClass
confusions$up$byClass
confusions$down$byClass
confusions$ROSE$byClass
confusions$SMOTE$byClass
