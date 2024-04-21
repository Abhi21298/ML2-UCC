## Section 3- KNN, Logistic, LDA, QDA

library(MASS) # for LDA
## Question 1 - KNN
library(class) #knn
library(car)
library(ISLR)
set.seed(1)
z = iris[sample(1:nrow(iris)),]
z[,5]
plot(z[,1:2], col=c(1,2,4)[z[,5]], 
     pch=20, cex=2)

k = 5
n = nrow(z)
x = z[,1:2]
y = z$Species
  
idxs = sample(1:n, 100)
x.train = x[idxs,]
x.test = x[-idxs,]
y.train = y[idxs]
y.test = y[-idxs]


## fitting a KNN
knn_mod = knn(x.train, x.test, y.train, k)
knn_mod # straight away gives outputs
# for test data!

tb = table(knn_mod, y.test)
1 - sum(diag(tb))/sum(tb)
confusionMatrix(data=knn_mod, reference=y.test)$overall[1]

## finding a better value for K! loop through it!
acc = numeric(30)*NA
for (k in 1:30) {
  knns = knn(x.train, x.test, y.train, k)
  acc[k] = confusionMatrix(data=knns, reference=y.test)$overall[1]
}
acc # best for k = 6

# plotting correct roc 
# knn gives preds of Pr(Y=Y_hat|X) not Pr(Y=Y|X)
# meaning provides probs of that class associated with the row
# gives probs of both 'Yes', 'No' in the preds columns
# not probs of 'Yes' alone!
# lot of conversion needed to be done
# for 2 class problem, 

# knn.o = knn(x[train,], x[-train,], y[train], k)
# knn.preds = as.numeric(knn.o == 'Up')
# knn.p = attributes(knn(x[train,], x[-train,], y[train], k, prob=TRUE))$prob
# new.probs = 1 - knn.p
# final.knn.preds = ifelse(knn.preds == 1,knn.p, new.probs)
# roc_knn = roc(y_test_true, final.knn.preds)

best_model_preds = knn(x.train, x.test, y.train, 6)
attributes(knn(x.train, x.test, y.train, 6, prob = TRUE))$prob

# here it's a 3 class problem, can't extract correct probs, very complicated.

## Question 2 - glm logistic
n = nrow(iris)
is = sample(1:n, size=n, replace=FALSE)
dat = iris[is,-c(3,4)] # shuffled version of the original set
# recode into 2-class problem:
dat$is.virginica = as.numeric(dat$Species=="virginica") 
dat$Species = NULL # "remove" this component
names(dat)

x.train = dat[is[0:100],]
x.test = dat[is[101:150],c(1,2)]
y.test = dat[is[101:150],3]
glm_bin = glm(is.virginica~., data=x.train, family = binomial)
predis = predict(glm_bin, newdata=x.test, type='response')
str(predis)
predis
final.preds = ifelse(predis > 0.5,1,0)
table(y.test,final.preds)


## Question 3 LDA - 
set.seed(4061)
df = iris
attach(df)
df$is.virginica = as.numeric(df$Species=="virginica")
df$Species = NULL

## LDA - all variables are normally distributed
## different classes have the same covariance matrix.

## use bartlett's test for checking equal variance across class within the same predictor variable
## use shapiro's test for normality distribution of data for each predictor variable (class wise separated).
## visualise shapiro test with boxplot! if the dist. is not skewed it's normal.

for (i in 1:4){
  print(bartlett.test(df[,i]~df$is.virginica)$p.value)
}
# Ho: p > 0.05, covariance of individual predictor is same.
# Ha: variances are not equal
# [1] 0.9414987 - Ho
# [1] 0.002526359 - Ha
# [1] 1.947747e-11 - Ha
# [1] 1.695941e-07 - Ha


## check normal distribution of each predictor variable across classes

for (i in 1:4){
  print(shapiro.test(df[which(df$is.virginica==0),i])$p.value)
}
## Ho: p > 0.05, normal distribution 
## Ha: p < 0.05, not normal
# [1] 0.02075718 - ha
# [1] 0.6462519 - ho
# [1] 2.823944e-10 - ha
# [1] 1.633939e-09 - ha

for (i in 1:4){
  print(shapiro.test(df[which(df$is.virginica==1),i])$p.value)
}
## same as above

## boxplot of the above test
par(mfrow=c(2,2))

for (i in 1:4){
  boxplot(df[,i]~df$is.virginica)
}

## lda
lda.model = lda(is.virginica~., data = df)
lda.model ## (DS: -0.19, 0.87, 0.017, 2.39)


## Interpretation for the model fit is that based on the discriminant
## score for the coefficients, Petal.Width plays the major role 
## in creating a linear boundary between the two classes, followed
## by Sepal.Width which takes the second most precendence. take absolute values


## Question 4 - QDA
## Assumptions for QDA changes w.r.t to LDA
library(MASS) ## for LDA
set.seed(4061)
df = iris
attach(df)
df$is.virginica = as.numeric(df$Species=="virginica")
df$Species = NULL
X = df
X$is.virginica = NULL
y = df$is.virginica
idx = sample(1:nrow(df), 100)
X.train = X[idx,]
X.test = X[-idx,]
y.train = y[idx]
y.test = y[-idx]

lda_model = lda(y.train ~., data = X.train)
lda_model
###
#Positive Coefficient:
#A positive coefficient means that as the value of that predictor variable increases, the corresponding linear discriminant value also increases.
#In terms of classification, this means that higher values of the predictor variable tend to push observations towards that class.
#Negative Coefficient:
#A negative coefficient means that as the value of that predictor variable increases, the corresponding linear discriminant value decreases.
#In terms of classification, this means that higher values of the predictor variable tend to push observations away from that class.
###

## lda predict
preds = predict(lda_model, newdata=X.test)
preds$class
table(y.test, preds$class)

## qda 
qda_model = qda(y.train~., data=X.train)
qda_preds = predict(qda_model, newdata= X.test)
table(y.test, qda_preds$class)

########## Differences between LDA and QDA #############
#Covariance Matrix:
#LDA: Assumes all classes share the same covariance matrix.
#QDA: Allows each class to have its own covariance matrix.
#Decision Boundary:
#LDA: Linear decision boundary (straight line or plane).
#QDA: Quadratic decision boundary (curved surface).
#Flexibility:
#LDA: Less flexible, assumes linear relationships.
#QDA: More flexible, can capture non-linear relationships.
#Computational Efficiency:
#LDA: Generally more computationally efficient due to simpler assumptions.
#QDA: Can be more computationally expensive, especially with large datasets and many features.
