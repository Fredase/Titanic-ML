setwd("C:/Users/Ezinne/Desktop/TITANIC")
library(caret)
library(randomForest)
library(ggplot2)
library(dplyr)
library(caTools)
library(corrplot)
library(rpart)
library(rpart.plot)
library(e1071)
library(ROCR)
library(pROC)
library(grid)
library(glmnet)

##import the data set
train <- read.csv("train.csv", stringsAsFactors = FALSE)
test <- read.csv("test.csv", stringsAsFactors = FALSE)
str(train)
str(test)

##The test data set has no "Survived" column so we set it as NA. also we track the respective entries 
##from train and set
test$Survived <- NA
train$set <- "train"
test$set <- "test"

##Bind the two data sets
full.set <- rbind(train, test)

##Check for unique values in each column
lapply(full.set, function(x) length(unique(x))) 

##Check for missing values
mv <- full.set %>% summarize_all(funs(sum(is.na(.))/n()))
mv <- tidyr::gather(mv, key = "feature", value = "Percent Missing")
mv %>% ggplot(aes(reorder(feature, -`Percent Missing`), `Percent Missing`)) +
  geom_bar(stat = "identity", fill = "red") +
  coord_flip() + theme_bw() +
  labs(x = "Feature", y = "Percent Missing") +
  ggtitle("Percentage of Missing values per feature")

###Useful data quality function for missing values
ColumnEval <- function(df, col){
  
  testcol <- df[[col]]
  numMissing = max(sum(is.na(testcol)| is.nan(testcol) | testcol == ''), 0)
  
  if (class(testcol) == 'numeric' | class(testcol) == 'Date' | class(testcol) == 'difftime' | class(testcol) == 'integer'){
    
    list('col' = col,'class' = class(testcol), 
         'num' = length(testcol) - numMissing, 'numMissing' = numMissing, 
         'numInfinite' = sum(is.infinite(testcol)), 'avgVal' = mean(testcol, na.rm=TRUE), 
         'minVal' = round(min(testcol, na.rm = TRUE)), 'maxVal' = round(max(testcol, na.rm = TRUE)))
  } else{
    list('col' = col,'class' = class(testcol), 
         'num' = length(testcol) - numMissing, 
         'numMissing' = numMissing, 'numInfinite' = NA,  
         'avgVal' = NA, 'minVal' = NA, 'maxVal' = NA)
    
  }
 
}
CheckColumns <- function(df){
  
  resDF <- data.frame()
  for(colName in names(df)){
    resDF = rbind(resDF, as.data.frame(ColumnEval(df = df, col = colName)))
  }
  
  resDF
}

CheckColumns(full.set)

percent_missing <- purrr::map_dbl(full.set, function(x) { round((sum(is.na(x)) / length(x)) * 100, 1) })
precent_missing <- percent_missing[percent_missing > 0]
data.frame(Miss = percent_missing, Var = names(percent_missing), row.names = NULL) %>%
  ggplot(aes(reorder(Var, -Miss), Miss)) + 
  geom_bar(stat = "identity", fill = "darkred") +
  labs(x = "", y = "%Missing") + ggtitle("Percent Missing by Feature") +
  theme(axis.text.x=element_text(angle=90, hjust=1))
###FEATURE ENGINEERING

table(full.set$Embarked) ##This shows that there are two observations imputed with an empty string
##Replace missing values in "Embarked" with the mode
full.set$Embarked[full.set$Embarked == ""] <- "S"
## Alternatively apply full.set$Embarked <- replace(full.set$Embarked, which(is.na(full.set$Embarked)), 'S')

##Check the number of missing values in Age feature
sum(is.na(full.set$Age))
full.set$MissingAge <- ifelse(is.na(full.set$Age), "Y", "N")
##Impute missing values in the Age variable by replacing NA with the mean Age
full.set <- full.set %>% mutate(Age = ifelse(is.na(Age), mean(full.set$Age, na.rm = TRUE), Age))

##Add new feature for Family size
full.set$FamilySize <- 1 + full.set$SibSp + full.set$Parch

#Convert some variables to factors
full.set$Survived <- as.factor(full.set$Survived)
full.set$Pclass <- as.factor(full.set$Pclass)
full.set$Sex <- as.factor(full.set$Sex)
full.set$Embarked <- as.factor(full.set$Embarked)
full.set$MissingAge <- as.factor(full.set$MissingAge)

##A look at the 'Survived 'Feature
table(full.set$Survived)
##Make a contingency table for Survived feature
surv_summary <- full.set %>% filter(set == "train") %>% select(PassengerId, Survived) %>%
  group_by(Survived) %>% summarise(n = n()) %>% mutate(freq = n/sum(n))
View(surv_summary)
surv_rate <- surv_summary$freq[surv_summary$Survived == "1"]

##   EDA    ##
##Explore the relationship between the Dependent and various Predictor variables
##Pclass
ggplot(full.set %>% filter(set == "train"), aes(Pclass, fill = Survived)) +
  geom_bar(position = "fill") + scale_fill_brewer(palette="Set1") +
   ylab("Survival Rate") +
  geom_hline(yintercept = surv_rate, col = "white", lty = 2, size = 2) +
  ggtitle("Survival Rate by Class") + theme_minimal()

##Sex
ggplot(full.set %>% filter(set == "train"), aes(Sex, fill = Survived)) +
  geom_bar(position = "fill") + scale_fill_brewer(palette = "Set1") +
  ylab("Survival Rate") +
  geom_hline(yintercept = surv_rate, col = "white", lty = 2, size = 2) +
  ggtitle("Survival Rate by Sex") +  theme_minimal()

##Age
age_tbl <- full.set %>% filter(set == "train") %>%
  select(Age, Survived) %>% group_by(Survived) %>%
  summarise(mean.age = mean(Age, na.rm=TRUE))

ggplot(full.set %>% filter(set == "train"), aes(Age, fill = Survived)) +
  geom_histogram(aes(y = ..density..), alpha = 0.5) + 
  geom_density(alpha = 0.2, aes(col = Survived)) +
  geom_vline(data = age_tbl, aes(xintercept = mean.age, col = Survived), lty = 2, size = 1) +
  scale_fill_brewer(palette = "Set1") + scale_colour_brewer(palette = "Set1") +
  ylab("Density") + ggtitle("Survival Rate by Age") 
  

##SibSp
ggplot(full.set %>% filter(set == "train"), aes(SibSp, fill = Survived)) +
  geom_bar(position = "fill") + scale_fill_brewer(palette="Set1") +
  geom_hline(yintercept = surv_rate, col = "white", lty = 2, size = 2) +
  ylab("Survival Rate") + ggtitle("Survival Rate by SibSp") + theme_minimal()

##Parch
ggplot(full.set %>% filter(set == "train"), aes(Parch, fill = Survived)) +
  geom_bar(position = "fill") + scale_fill_brewer(palette = "Set1") +
  ylab("Survival Rate") + 
  geom_hline(yintercept = surv_rate, col = "white", lty = 2, size = 2) +
  ggtitle("Survival Rate by Parch") + theme_bw()
  

##Embarked
ggplot(full.set %>% filter(set == "train"), aes(Embarked, fill = Survived)) +
  geom_bar(position = "fill") + scale_fill_brewer(palette = "Set1") +
  geom_hline(yintercept = surv_rate, col = "white", lty = 2, size = 2) +
  ylab("Survival Rate") + ggtitle("Survival Rate by Embarked") + theme_bw()

##Family Size
ggplot(full.set %>% filter(set == "train") %>% na.omit, aes(FamilySize, fill = Survived)) +
  geom_bar(position = "fill") + scale_fill_brewer(palette = "Set1") +
  ylab("Survival Rate") + 
  geom_hline(yintercept = surv_rate, col = "white", lty = 2, size = 2) +
  ggtitle("Survival Rate by Family Size") + theme_bw() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

##Relationships by Frequency

##Pclass
ggplot(full.set %>% filter(set == "train"), aes(Pclass, fill = Survived)) +
  geom_bar(position = "stack") + scale_fill_brewer(palette = "Set1") +
  ylab("Passengers") + ggtitle("Survived by Class") + theme_minimal()
  

##Sex
ggplot(full.set %>% filter(set == "train"), aes(Sex, fill = Survived)) +
  geom_bar(position = "stack") + scale_fill_brewer(palette = "Set1") +
  ylab("Passengers") + ggtitle("Survived by Sex") + theme_minimal()

##Age
ggplot(full.set %>% filter(set == "train"), aes(Age, fill = Survived)) +
  geom_histogram(aes(y = ..count..), alpha = 0.5) +
  geom_vline(data = age_tbl, aes(xintercept = mean.age, col = Survived), lty = 2, size = 1) +
  scale_fill_brewer(palette = "Set1") + scale_colour_brewer(palette = "Set1") +
  ylab("Frequency") + ggtitle("Survived by Age") + theme_bw()
  

##SibSp
ggplot(full.set %>% filter(set == "train"), aes(SibSp, fill = Survived)) +
  geom_bar(position = "stack") + scale_fill_brewer(palette = "Set1") +
  ylab("Passengers") + ggtitle("Survived by SibSp") + theme_minimal()


##Parch
ggplot(full.set %>% filter(set == "train"), aes(Parch, fill = Survived)) +
  geom_bar(position = "stack") + scale_fill_brewer(palette = "Set1") +
  ylab("Passengers") + ggtitle("Survived by Parch") + theme_minimal()

##Embarked
ggplot(full.set %>% filter(set == "train"), aes(Embarked, fill = Survived)) +
  geom_bar(position = "stack") + scale_fill_brewer(palette = "Set1") +
  ylab("Passengers") + ggtitle("Survived by Embarked") + theme_minimal()

##Family Size
ggplot(full.set %>% filter(set == "train") %>% na.omit, aes(FamilySize, fill = Survived)) +
  geom_bar(position = "stack") + scale_fill_brewer(palette = "Set1") +
  ylab("Passengers") + ggtitle("Survived by Family Size") + theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

##Investigate interactive relationships between variables
##Correlation

tbl_corr <- full.set %>% filter(set == "train") %>%
  select(-PassengerId, -SibSp, -Parch) %>%
  select_if(is.numeric) %>% cor(use="complete.obs") %>%
  corrplot.mixed(tl.cex = 1.5)

##Mosaic plot
mosaic_tbl <- full.set %>% filter(set == "train") %>%
  select(Survived, Pclass, Sex, Embarked, FamilySize) %>% mutate_all(as.factor)

mosaicplot(~ Pclass + Sex + Survived, data = mosaic_tbl, shade = TRUE,
           type = "pearson", main = "Mosaic Plot of numeric variables", color = TRUE)

##Alluvial
alluvial_tbl <- full.set %>%  filter(set=="train") %>%
  group_by(Survived, Sex, Pclass) %>%
  summarise(N = n()) %>% ungroup %>% na.omit

alluvial::alluvial(alluvial_tbl, freq = alluvial_tbl$N, border = NA,
                   col = ifelse(alluvial_tbl$Survived == "1", "blue", "gray"),
                   cex = 0.8,
                   ordering = list(order(alluvial_tbl$Survived, alluvial_tbl$Pclass == "1"),
                                   order(alluvial_tbl$Sex, alluvial_tbl$Pclass == "1"), NULL, NULL))

##Apply Machine Learning Algorithms
feature1 <- full.set[1:891, c("Survived", "Pclass", "Sex", "Age", "SibSp",
                             "Parch", "Fare", "Embarked", "MissingAge",
                             "FamilySize")]
depvar <- as.factor(train$Survived)

##For Cross Validation purposes, we will split our original train data by a 70/30 ratio
## This to check how well our model works

set.seed(54321)
indexes <- createDataPartition(feature1$Survived, times = 1, p = 0.7, list = FALSE )
train_var <- feature1[indexes,]
test_var <- feature1[-indexes,]

##Examine the proportion of the Survived label across split data
prop.table(table(train$Survived))
prop.table(table(train_var$Survived))
prop.table(table(test_var$Survived))

## Now to apply different ML algorithms
##Decision Tress
set.seed(12345)
DT.Model <- rpart(Survived ~ ., data = train_var, method = "class")
rpart.plot(DT.Model, extra = 5, fallen.leaves = TRUE)
##Make prediction using this model

DT.predict <- predict(DT.Model, data = train_var, type = "class")

##Use a confusion matrix to check accuracy of the predictions
confusionMatrix(DT.predict, train_var$Survived)

##There is a chance of overfitting so we apply a 10-fold cross validation
set.seed(12345)

cv <- createMultiFolds(train_var$Survived, k = 10, times = 10)
train.control <- trainControl(method = "repeatedcv", number = 10,
                              repeats = 10, index = cv)

CVT.Model <- train(x = train_var[, -1], y = train_var[, 1], method = "rpart",
                   tuneLength = 30, trControl = train.control)
print(CVT.Model)
rpart.plot(CVT.Model$finalModel, extra = 5, fallen.leaves = TRUE)

##Our model accuracy is approximately 0.80, with Sex, Age, Pclass and Family size being the more important
##variables
##Lets cross-validate the accuracy using the test data
CVT.predict <- predict(CVT.Model$finalModel, newdata = test_var, type = "class")
confusionMatrix(CVT.predict, test_var$Survived)
##The acccuracy of the model on the test data is 0.8195

###RANDOM FOREST

set.seed(12345)

RF1 <- randomForest(x = train_var[,-1], y = train_var[, 1], importance = TRUE, ntree = 1000)
RF1 ##Accuracy is 82.24 ie (100 - error rate)
varImpPlot(RF1)

##Remove the four redundant variables 'Embarked', "SibSp', 'Parch' and 'MissingAge' from train and test sets
train_var1 <- train_var[,-c(5:6, 8:9)]
test_var1 <- test_var[, c(5:6, 8:9)]

set.seed(12345)

RF2 <- randomForest(x = train_var1[,-1], y = train_var1[, 1], importance = TRUE, ntree = 1000)
RF2 ##Surprisingly the accuracy of the model reduces to 80.64%
#We revert to the first model for cross validation
set.seed(54321)
cv1 <- createMultiFolds(train_var[, 1], k = 10, times = 10)

ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10, index = cv1)

RF3 <- train(x = train_var[, -1], y = train_var[, 1], method = "rf",
             tuneLength = 5, ntree = 1000, trControl = ctrl)
print(RF3)
##Cross-Validation gave us an accuracy rate of 0.8217 or 82.17%
##Make predictions with test data

test.pred <- predict(RF3, newdata = test_var)

##Check accuracy with Confusion Matrix
confusionMatrix(test.pred, test_var$Survived) ##Gives an accuracy rate of 0.8383, better than expected

##LASSO-RIDGE REGRESSION
train.male = subset(train_var, train_var$Sex == "male")
train.female = subset(train_var, train_var$Sex == "female")
test.male = subset(test_var, test_var$Sex == "male")
test.female = subset(test_var, test_var$Sex == "female")

train.male$Sex = NULL
train.female$Sex = NULL
test.male$Sex = NULL
test.female$Sex = NULL

##Split the train set 
set.seed(105)
train_ind <- sample.split(train.male$Survived, SplitRatio = .75)

##MALE
##set seed for reproducibility
set.seed(101)
cv.train.m <- train.male[train_ind, ]
cv.test.m  <- train.male[-train_ind, ]

##FEMALE
set.seed(105)
train_ind1 <- sample.split(train.female$Survived, SplitRatio = .75)

set.seed(101)
cv.train.f <- train.female[train_ind1, ]
cv.test.f <- train.female[-train_ind1, ]

##10-fold Cross-validation
set.seed(456)
x.m <- data.matrix(cv.train.m[, -1])
y.m <- cv.train.m$Survived
cvfit.m.ridge <- cv.glmnet(x.m, y.m, family = "binomial",
                           alpha = 0, type.measure = "class")

cvfit.m.lasso <- cv.glmnet(x.m, y.m, family = "binomial",
                           alpha = 1, type.measure = "class")

par(mfrow=c(1,2))
plot(cvfit.m.ridge, main = "Ridge")
plot(cvfit.m.lasso, main = "Lasso")

##Extract the coefficients of the models
coef(cvfit.m.ridge, s = "lambda.min")

##Predict on the training set
PredTrainMale <- predict(cvfit.m.ridge, newx = x.m, type = "class")
table(cv.train.m$Survived, PredTrainMale)

##Make prediction on the test set
PredTestMale = predict(cvfit.m.ridge, newx = data.matrix(test.male[,-1]), type = "class")

summary(PredTestMale)

##FEMALE
set.seed(101)
x.f <- data.matrix(cv.train.f[, -1])
y.f <- cv.train.f$Survived

set.seed(456)

cvfit.f.ridge <- cv.glmnet(x.f, y.f, family = "binomial",
                           alpha = 0, type.measure = "class")

cvfit.f.lasso <- cv.glmnet(x.f, y.f, family = "binomial",
                           alpha = 1, type.measure = "class")

par(mfrow=c(1,2))
plot(cvfit.f.ridge, main = "Ridge")
plot(cvfit.f.lasso, main = "Lasso")

coef(cvfit.f.ridge, s = "lambda.min")
##Prediction usin the Ridge model

PredTrainFemale <- predict(cvfit.f.ridge, newx = x.f, type = "class")
table(cv.train.f$Survived, PredTrainFemale)
confusionMatrix(cv.train.f$Survived, PredTrainFemale) ##0.7939 accuracy rate

##Prediction on Validation(test) set
PredTestFemale = predict(cvfit.f.ridge, newx = data.matrix(cv.test.f[, -1]), type = "class")
table(cv.test.f$Survived, PredTestFemale)
confusionMatrix(cv.test.f$Survived, PredTestFemale) ##0.7991 accuracy rate

##Prediction using the Lasso Model
PredTrain.F = predict(cvfit.f.lasso, newx = x.f, type = "class")
table(cv.train.f$Survived, PredTrain.F)
confusionMatrix(cv.train.f$Survived, PredTrain.F) ##0.7939 accuracy rate

##Prediction on the Validation set
PredTest.F = predict(cvfit.f.lasso, newx = data.matrix(cv.test.f[, -1]), type="class")
table(cv.test.f$Survived, PredTest.F)
confusionMatrix(cv.test.f$Survived, PredTest.F) ##0.7991 accuarcy rate

##Gather the results
SubFemale <- cbind(cv.train.f$Survived, PredTrain.F)
SubMale <- cbind(cv.train.m$Survived, PredTrainMale)

Sub <- rbind(SubMale, SubFemale)

colnames(Sub) <- c("Actual", "Predicted")
Sub <- as.data.frame(Sub)
##The 'Actual' variable has different levels than the "Predicted' variable
## Make the levels to be same
levels(Sub$Actual)[levels(Sub$Actual) == "1"] <- "0"
levels(Sub$Actual)[levels(Sub$Actual) == "2"] <- "1"

confusionMatrix(Sub$Actual, Sub$Predicted) ##This gives a 0.8013 accuracy rate

##LINEAR SUPPORT VECTOR MACHINE

##First tune the Cost Parameter using tune.svm() from the e1071 package
linear.tune <- tune.svm(Survived ~ ., data = train_var, kernel = "linear",
                        cost=c(0.01,0.1,0.2,0.5,0.7,1,2,3,5,10,15,20,50,100))
print(linear.tune)
##Best Performance is Cost = 10, and accuracy = 0.7825

##Extract the best linear model
Best.Linear <- linear.tune$best.model
##Predict Survival the test data
BestTestPred <- predict(Best.Linear, newdata = test_var, type = "class")
confusionMatrix(test_var$Survived, BestTestPred) ## Accuracy rate of 0.8045 

##NON-LINEAR SVM
set.seed(12345)

tuner <- tune.svm(Survived ~ ., data = train_var, kernel = "radial", gamma = seq(0.1, 5))

summary(tuner) ##The Non Linear Kernel gives us better accuracy of 0.8047

best.Tuner <- tuner$best.model
##Make predictions on test data
tune.pred <- predict(best.Tuner, newdata = test_var)
confusionMatrix(tune.pred, test_var$Survived)
#The accuracy of the non-linear model is much improved at 84.59% compared to the linear model

##LOGISTIC REGRESSION
contrasts(train_var$Sex)
contrasts(train_var$Pclass)
##The contrasts function shows how a factor variable compares with itself
##Run the Logistic model
log.mod <- glm(Survived ~ ., family = binomial(link=logit), data = train_var)

summary(log.mod)
confint(log.mod)

##Predict the train data
train.prob <- predict(log.mod, newdata = train_var, type = "response")
table(train_var$Survived, train.prob > 0.5)
(337 + 167)/(337 + 73 + 48 + 167) ##This yields accuracy rate of 0.8064
##That is to say the Logistic model predicted the train data with 80.64% accuracy

##Predict on the test data
test.prob <- predict(log.mod, newdata = test_var,type = "response")
table(test_var$Survived, test.prob > 0.5)

(143 + 70)/(143 + 32 + 21 + 70) ##This yields accuracy rate of 0.8007519
##The accuracy of the logistic model on the test data is 80.1%