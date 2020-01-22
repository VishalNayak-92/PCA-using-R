# Remove all the environment variables
rm(list=ls(all=TRUE))

# Set working directory
setwd("~/Downloads/201900922_Batch70_CSE7402c_PCA/")
#directory where you have kept your data


# Read the data
data<-read.csv("housing_data.csv", header=T)
str(data)
summary(data)

#* The column/variable names' explanation is given below:

# 1) __CRIM :__ Per capita Crime rate by town
# 2) __ZN :__ Proportion of residential land zoned for lots over 25,000 sq.ft.
# 3) __INDUS :__ Proportion of non-retail business acres per town
# 4) __CHAS :___ Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 5) __NOX :__ nitric oxides concentration (parts per 10 million)
# 6) __RM :__ average number of rooms per dwelling
# 7) __AGE :__ proportion of owner-occupied units built prior to 1940
# 8) __DIS :__ weighted distances to five Boston employment centres
# 9) __RAD :__ index of accessibility to radial highways
# 10) __TAX :__ full-value property-tax rate per $10,000
# 11) __PTRATIO :__ pupil-teacher ratio by town
# 12) __B :__ 1000(Bk - 0.63)^2 where Bk is the proportion of African-Americans by town
# 13) __LSTAT :__ Percentage of the population in the lower economic status
# 14) __MV  :__ Median value of owner-occupied homes in multiples of $1000



# Missing values
sum(is.na(data))

#In this problem we are trying to predict the 
#median value of house given the features. But we
#have missing values in the target. So there is no point
#in imputing them.
data<-data[!is.na(data$MV),]

str(data)

# Type Conversions
data$CHAS<-as.factor(as.character(data$CHAS))
data$RAD<-as.factor(as.character(data$RAD))
str(data)

#As we see there are missig values, we need 
#to impute them. But before that
#We need to split the data into train and test

# split data 

set.seed(123)
library(caret)
trainRows=createDataPartition(data$MV,p = 0.8,list = FALSE)

train=data[trainRows,]
validation=data[-(trainRows),]


colnames(train)

# Separating Independent and dependent variables
#Since we do not want to use target to predict
# missing values in independent variables
x_train<-train[,-14]
x_validation<-validation[,-14]
y_train<-train[,14]
y_validation<-validation[,14]

# Missing Value Imputation
library(DMwR)
train_full<-knnImputation(x_train)
validation_full<-knnImputation(x_validation,distData = x_train)

# Combine imputed predictors with target into a dataframe
train_full<-data.frame(train_full,Target=y_train)
validation_full<-data.frame(validation_full,Target=y_validation)

# Build a linear model on this data
linmod<-lm(Target~.,data=train_full)

linmod
pred_full<-predict(linmod,newdata=validation_full)

#Evaluate
error_lm_full = regr.eval(validation_full$Target,pred_full)
error_lm_full

#########PCA####
#We know that principal components are obtained on numeric
#data. From the data description, we understand that CHAS
#and RAD are categorical, hence we shall keep these two
#variables out which constructing principal components

train1<-subset(train_full,select=-c(CHAS,RAD,Target))
validation1<-subset(validation_full,select=-c(CHAS,RAD,Target))
str(train1)

# Standardization of the values before obtaining
# principal components. This is important as pca
# is variance maximization exercise
library(caret)
trainobj=preProcess(train1,method = "scale")
train1=predict(trainobj,train1)
validation1=predict(trainobj,validation1)



###  Alpplying PCA

pca_std = princomp(train1)

pca_std
# Loadings are the factors which show what amount of load is shared by a 
# particular feature.So, the component with maximum loading value can be 
# considered as the most influential feature

summary(pca_std)
names(pca_std)
pca_std$scores
train1

pca_train = predict(pca_std, train1)
pca_validation = predict(pca_std, validation1)



#Checking how many components are really required
plot(pca_std)

#biplot(pca_std)
screeplot(pca_std)
screeplot(pca_std, type="lines")


#Lets say 6 components
pca_X=data.frame(pca_train[,1:6],train_full[,c(4,9,14)])
pca_v=data.frame(pca_validation[,1:6],validation_full[,c(4,9,14)])


linreg<-lm(Target~.,data=pca_X)
pred_valid<-predict(linreg,newdata = pca_v)

library(DMwR)
error_pca_std=regr.eval(y_validation,pred_valid)

print(error_lm_full)
print(error_pca_std)

# Observe that lm on all the 13 attributes gave a mape of about 18% on the validation data
# lm after reducing dimensions to 6 (to half) gave a mape of about 19% mape on the validation set



#library(devtools)
#install_github("vqv/ggbiplot")
library(ggbiplot)

ggbiplot(pca_std)
