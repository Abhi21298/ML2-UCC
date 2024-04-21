## Section 5 - SVMs
# no need to convert model.matrix when ALL X columns numeric
# do it only if there are categorical variables in Xs. 
# basically model.matrix does one-hot encoding by itself! that's the key!
# By default, convert y to factor if not done already!

rm(list=ls())
# Question 1
library(pROC)
library(ISLR)
require(ISLR) # contains the dataset
# Recode response variable so as to make it a classification problem
High = ifelse(Carseats$Sales<=8, "No", "Yes")
CS = data.frame(Carseats, High)
CS$Sales = NULL
x = CS
x$High = NULL
y = CS$High

# split the data into train+test:
n = nrow(CS)
set.seed(4061)
i.train = sample(1:n, 350) 
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]
y.train

## svm model - problem (expects x as matrix)
## can't handle y as yes, no.. convert to factors!!
svmo = svm(x.train, y.train, kernel = 'polynomial')

xm = model.matrix(y~.+0, data=x)
xm.train = xm[i.train,]
xm.test = xm[-i.train,]
y = as.factor(y)
y.train = y[i.train]
y.test = y[-i.train]

svml = svm(xm.train, y.train, kernel = 'linear')
summary(svml)
svmp = svm(xm.train, y.train, kernel = 'polynomial')
summary(svmp)

# identify svm indexes
length(svml$index)
## identify svm coordinates
svml$SV
yl = as.numeric(y=='Yes') + 1
plot(apply(xm,2,scale), pch=c(15,20)[yl], col=c(1,4)[yl], 
     cex=c(1.2,2)[yl], main="The data")

plot(apply(xm.train,2,scale), pch=c(15,20)[yl[i.train]], col=c(1,4)[yl[i.train]], 
     cex=c(1.2,2)[yl[i.train]], main="linear")
points(svml$SV, pch = 5, col = 2, cex = 1.2)
plot(apply(xm.train,2,scale), pch=c(15,20)[yl[i.train]], col=c(1,4)[yl[i.train]], 
     cex=c(1.2,2)[yl[i.train]], main="polynomial")
points(svmp$SV, pch = 5, col = 2, cex = 1.2)

## fit accuracy
fitted_linear = fitted(svml)
fitted_poly = fitted(svmp)
table(y.train, fitted_linear)
table(y.train, fitted_poly) ## better accuracy
summary(svmp)
summary(svml)

## testing predictions
pred1 = predict(svml, newdata=xm.test) ## linear kernel works better
pred2 = predict(svmp, newdata=xm.test)
table(y.test, pred1)
table(y.test, pred2)

## for roc we need probs - probability - true while fitting itself!
svml_new = svm(xm.train, y.train, kernel = 'linear', probability = TRUE)
svmp_new = svm(xm.train, y.train, kernel = 'polynomial', probability = TRUE)


pred_lin = predict(svml_new, newdata= xm.test, probability=TRUE)
pred_pol = predict(svmp_new, newdata= xm.test, probability=TRUE)
## conf matrix
caret::confusionMatrix(data = pred_lin, reference = y.test, positive = 'Yes')
caret::confusionMatrix(data = pred_pol, reference = y.test, positive = 'Yes')
# retrieve yes probs
lin_probs = attributes(pred_lin)$probabilities[,2]
# convert y as factors for roc analysis
y.test = as.factor(y.test)

?roc
rocl = roc(response=y.test, predictor = lin_probs)
plot(rocl)

## Question 2 - iris data
x = iris
x$Species = NULL
y = iris$Species

set.seed(4061)
n = nrow(x)
i.train = sample(1:n, 100) 
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]
par(mfrow=c(1,1))
# (a) 
plot(x.train[,1:2], pch=20, col=c(1,2,4)[as.numeric(y.train)], cex=2)

# (b)
dat = data.frame(x.train, y=as.factor(y.train))
svmo.lin = svm(y~., data=dat, kernel='linear') 
svmo.pol = svm(y~., data=dat, kernel='polynomial') 
svmo.rad = svm(y~., data=dat, kernel='radial') 
#
# number of support vectors:
summary(svmo.lin)
summary(svmo.pol)
summary(svmo.rad)
#
# test error:
pred.lin = predict(svmo.lin, newdata=x.test)
pred.pol = predict(svmo.pol, newdata=x.test)
pred.rad = predict(svmo.rad, newdata=x.test)
cm.lin = confusionMatrix(y.test, pred.lin)
cm.pol = confusionMatrix(y.test, pred.pol)
cm.rad = confusionMatrix(y.test, pred.rad)
c(cm.lin$overall[1], cm.pol$overall[1], cm.rad$overall[1])

## tuning for radial function
tune_paras = e1071::tune(svm, train.x = x.train, train.y = y.train, kernel = 'radial', ranges = list(cost=10^c(-2:2), gamma = c(0.5,1,1.5,2)))
tune_paras
tune_paras$best.parameters                         

#3 tuned_svm_rad
tuned_svm_rad = svm(y~., data=dat, kernel='radial',
                    gamma=tune_paras$best.parameters$gamma,
                    cost=tune_paras$best.parameters$cost)
tune_pred = predict(tuned_svm_rad)
table(tune_pred, y.train) #96% lower than 100% for untuned radial but
# still comparable results! not an disadvantage!

## Question 3

set.seed(4061)
library(caret)

dat = Hitters
dat = na.omit(dat)
n = nrow(dat)
dat$Salary = as.factor(ifelse(dat$Salary>median(dat$Salary),"High","Low"))

idxs= sample(1:n, round(0.7*n))
dat.train = dat[idxs,]
dat.test = dat[-idxs,]
y.test.true = dat.test$Salary
dat.test$Salary = NULL
x.train = dat.train
y.train = dat.train$Salary
x.train$Salary = NULL

## svm using caret
svmL_caret = train(Salary~.,data = dat.train, method='svmLinear')
preds = predict(svmL_caret, dat.test)
preds
confusionMatrix(preds, y.test.true)
# rest all are same

## Question 4 - SVM Regression
x = iris
x$Sepal.Length = NULL
y = iris$Sepal.Length
pairs(iris[,1:4])

set.seed(4061)
n = nrow(x)
i.train = sample(1:n, 100) 
x.train = x[i.train,]
x.test = x[-i.train,]
y.train = y[i.train]
y.test = y[-i.train]
dat.train = cbind(x.train,y=y.train)

## caret along with CV 

train_par = trainControl(method = "cv")
svm_radial_regr = caret::train(y~.,data=dat.train, method = "svmRadial",
             trControl = train_par)
preds = predict(svm_radial_regr, x.test)
mean((preds-y.test)^2)



set.seed(4061)

data = iris
data$Species = NULL
n = nrow(data)
X = data
y = data$Sepal.Length
X$Sepal.Length = NULL

idx = sample(1:n, 100)

X.train = X[idx, ]
X.test = X[-idx, ]
y.train = y[idx]
y.test = y[-idx]

train.data = data.frame(X.train, y.train)
train_control = trainControl(method = "cv", number = 10)
model = train(y.train~., data = train.data, method = "svmRadial", trControl = train_control)
model

svm.p = predict(model, newdata=X.test)
mean( (y.test-svm.p)^2 )
