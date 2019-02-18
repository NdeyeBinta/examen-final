



## Importing packages
library(plyr)
library(tidyverse) 
library(MASS)
library(car)
library(e1071)
library(caret)
library(cowplot)
library(caTools)
library(pROC)
library(ggcorrplot)
library(reshape)

#========================================WHAT'S ABOUT DATASET===================================
list.files(path = "../input")
# Load and view the dataset
abone <- read.csv("films.csv",sep=";")
View(abone)
abone$X<-NULL
abone$X.1<-NULL
abone$X.2<-NULL
abone$X.3<-NULL
abone$X.4<-NULL
abone$X.5<-NULL
# See the variables types
glimpse(abone)
str(abone)



#===========================Take a look to missing data in telco dataset================================= 
# Take the percent of the count missing data in each column
missing_data <- abone %>% summarise_all(funs(sum(is.na(.))/n()))
View(missing_data)

# Transform it as a dataframe with 2 columns 
missing_data <- gather(missing_data, key = "variables", value = "percent_missing")


# Create a complete dataset without NAs
abone_full <- abone[complete.cases(abone),]
glimpse(abone_full)
str(abone_full)

abone_int <- abone_full[,c("desa", "annee", "prix")]

abone_cat <- abone_full[,-c(1,6,19,20)]
#Creating Dummy Variables
dummy<- data.frame(sapply(abone_cat,function(x) data.frame(model.matrix(~x-1,data =abone_cat))[,-1]))
head(dummy)
# Create the final dataset by combining the data
# Combining the data
abone_final <- cbind(abone_int,dummy)
head(abone_final)

#Splitting the data
set.seed(123)
indices = sample.split(abone_final$desa, SplitRatio = 0.7)
train = abone_final[indices,]
test = abone_final[!(indices),]

#=============================== Exploratory Modeling ======================================
# First Model building-logistic regression
model_1 = glm(desa ~ ., data = train, family = "binomial")
summary(model_1)
#
model_2<- stepAIC(model_1, direction="both")
summary(model_2)
vif(model_2)


#Model_3 all has significant variables, so let's just use it for prediction first
final_model <- model_1
#Prediction
pred <- predict(final_model, type = "response", newdata = validation[,24])
summary(pred)
validation$prob <- pred
# Using probability cutoff of 50%.
pred_desa <- factor(ifelse(pred >= 0.50, "Yes", "No"))
actual_desa <- factor(ifelse(validation$desa==1,"Yes","No"))
table(actual_desa,pred_desa)

perform_fn <- function(cutoff) 
{
  predicted_desa <- factor(ifelse(pred >= cutoff, "Yes", "No"))
  conf <- confusionMatrix(predicted_desa, actual_desa, positive = "Yes")
  accuray <- conf$overall[1]
  sensitivity <- conf$byClass[1]
  specificity <- conf$byClass[2]
  out <- t(as.matrix(c(sensitivity, specificity, accuray))) 
  colnames(out) <- c("sensitivity", "specificity", "accuracy")
  return(out)
}

options(repr.plot.width =8, repr.plot.height =6)
summary(pred)
s = seq(0.01,0.80,length=100)
OUT = matrix(0,100,3)

for(i in 1:100)
{
  OUT[i,] = perform_fn(s[i])
} 

plot(s, OUT[,1],xlab="Cutoff",ylab="Value",cex.lab=1.5,cex.axis=1.5,ylim=c(0,1),
     type="l",lwd=2,axes=FALSE,col=2)
axis(1,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
axis(2,seq(0,1,length=5),seq(0,1,length=5),cex.lab=1.5)
lines(s,OUT[,2],col="darkgreen",lwd=2)
lines(s,OUT[,3],col=4,lwd=2)
box()
legend("bottom",col=c(2,"darkgreen",4,"darkred"),text.font =3,inset = 0.02,
       box.lty=0,cex = 0.8, 
       lwd=c(2,2,2,2),c("Sensitivity","Specificity","Accuracy"))
abline(v = 0.32, col="red", lwd=1, lty=2)
axis(1, at = seq(0.1, 1, by = 0.1))

#cutoff <- s[which(abs(OUT[,1]-OUT[,2])<0.01)]

cutoff_desa <- factor(ifelse(pred >=0.32, "Yes", "No"))
conf_final <- confusionMatrix(cutoff_desa, actual_desa, positive = "Yes")
accuracy <- conf_final$overall[1]
sensitivity <- conf_final$byClass[1]
specificity <- conf_final$byClass[2]
accuracy
sensitivity
specificity


# Second model building-Decision Tree
set.seed(123)
desa_final$Churn <- as.factor(desa_final$Churn)
indices = sample.split(desa_final$Churn, SplitRatio = 0.7)
train = desa_final[indices,]
validation = desa_final[!(indices),]

# Training the decision tree using all variables
options(repr.plot.width = 10, repr.plot.height = 8)
library(rpart)
library(rpart.plot)
# Training 
Dtree = rpart(Churn ~., data = train, method = "class")
summary(Dtree)
# Predicting 
DTPred <- predict(Dtree,type = "class", newdata = validation[,-24])
#Checking the confusion matrix
confusionMatrix(validation$desa, DTPred)

# Third model building-Random forest
library(randomForest)
set.seed(123)
desa_final$Churn <- as.factor(desa_final$Churn)

indices = sample.split(desa_final$desa, SplitRatio = 0.7)
train = desa_final[indices,]
validation = desa_final[!(indices),]
#Training the RandomForest Model
model.rf <- randomForest(desa ~ ., data=train, proximity=FALSE,importance = FALSE,
                         ntree=500,mtry=4, do.trace=FALSE)
model.rf
#Predicting on the validation set and checking the Confusion Matrix.
testPred <- predict(model.rf, newdata=validation[,-24])
table(testPred, validation$desa)

confusionMatrix(validation$desa, testPred)
#Checking the variable Importance Plot
varImpPlot(model.rf)
options(repr.plot.width =10, repr.plot.height = 8)

glm.roc <- roc(response = validation$desa, predictor = as.numeric(pred))
DT.roc <- roc(response = validation$desa, predictor = as.numeric(DTPred))
rf.roc <- roc(response = validation$desa, predictor = as.numeric(testPred))

plot(glm.roc,      legacy.axes = TRUE, print.auc.y = 1.0, print.auc = TRUE)
plot(DT.roc, col = "blue", add = TRUE, print.auc.y = 0.65, print.auc = TRUE)
plot(rf.roc, col = "red" , add = TRUE, print.auc.y = 0.85, print.auc = TRUE)
legend("bottom", c("Random Forest", "Decision Tree", "Logistic"),
       lty = c(1,1), lwd = c(2, 2), col = c("red", "blue", "black"), cex = 0.75)

#Summary of all models


