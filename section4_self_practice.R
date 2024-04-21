## section 4 practice for ML 2

## Question 1

library(tree)
library(ISLR)
High = ifelse(test = Carseats$Sales<=8, "No", "Yes")
CS = data.frame(Carseats, High)
CS
# preprocessing required data
CS$Sales = NULL
str(CS)
CS$High = as.factor(CS$High)

# classification tree & plotting it
class_tree = tree(High~., data=CS)
summary(class_tree)

plot(class_tree)
text(class_tree, pretty=0)

# pruning a tree
pruned_tree = cv.tree(class_tree, FUN = prune.misclass)
summary(pruned_tree)
# choose the best value of size (number of terminal nodes) that
# resulted in the lowest deviance (error).
best_size = pruned_tree$size[which.min(pruned_tree$dev)]
final_prune = prune.misclass(class_tree, best = best_size)
final_prune
plot(final_prune)
text(final_prune, pretty = 0)
plot(class_tree)
text(class_tree, pretty = 0)


## Question 2 - similar to 1 but train test split
set.seed(2)
par(mfrow=c(1,1))
n = nrow(CS)
idx = sample(1:n, n/2, replace=FALSE)
length(idx)
train_X = CS[idx,]
train_Y = High[idx]
test_X = CS[-idx,]
test_Y = High[-idx]

tree.out.train = tree(High~., data=train_X)
summary(tree.out.train)
plot(tree.out.train)
text(tree.out.train, pretty = 0)
# training error
predictions = predict(tree.out.train, train_X, type = "class")
predictions
matrix1 = table(train_Y, predictions)
matrix1
sum(diag(matrix1))/sum(matrix1)

# comparing with previous pruned tree - training error
prune_pred = predict(final_prune, train_X, type = "class")
matrix2 = table(train_Y, prune_pred)
matrix2
sum(diag(matrix2))/sum(matrix2)

# comparison with test data

test_data_predictions = predict(tree.out.train, test_X, type = 'class')
(confu = table(test_Y, test_data_predictions))
sum(diag(confu))/sum(confu)

## ROC Curve
library(pROC)
# here we specify 'type="vector"' to retrieve continuous scores
# as opposed to predicted labels, so that we can apply varying
# threshold values to these scores to build the ROC curve:
ftree.probs = predict(tree.out.train, test_X, type="vector")
roc.f = roc(response=(test_Y), predictor=ftree.probs[,1])
roc.f$auc
plot(roc.f)


## Question 3 - unsuccessful attempt - what am i missing here?

# Hitters dataset - Regression tree now
Hitters
par(mfrow=c(1,2))
dat = Hitters
dat
dat$Salary = log(dat$Salary)
str(dat)
dat = na.omit(dat)
hitters.tree = tree(Salary~.,dat)
summary(hitters.tree)
plot(hitters.tree)
text(hitters.tree, pretty = 0)

prune.hitters = cv.tree(hitters.tree)
summary(prune.hitters)
best_size_hit = prune.hitters$size[which.min(prune.hitters$dev)]
final_prune_hitters = prune.tree(hitters.tree, best = best_size_hit)
plot(final_prune_hitters)
text(final_prune_hitters, pretty = 0)

## Question 4 - BAsic random forest
library(randomForest)
High = ifelse(test = Carseats$Sales<=8, "No", "Yes")
CS = data.frame(Carseats, High)
CS
# preprocessing required data
CS$Sales = NULL
str(CS)
CS$High = as.factor(CS$High)

class_tree = tree(High~., data=CS)
summary(class_tree)

plot(class_tree)
text(class_tree, pretty=0)
tree.preds = predict(class_tree, CS, type = "class")

# Growing a RF now with default params
raf.tree = randomForest(High~.,CS)
raf.tree.preds = predict(raf.tree, CS, type="class")

# Growing a RF with bagging mtry = number of columns - 1
P = ncol(CS) - 1
raf.tree.bag = randomForest(High~.,CS,mtry=sqrt(P))
raf.tree.bag.preds = predict(raf.tree.bag, CS, type="class")
(A1 = (table(High, tree.preds)))
(A2 = (table(High, raf.tree.preds)))
(A3 = (table(High, raf.tree.bag.preds)))
sum(diag(A1))/sum(A1)
sum(diag(A2))/sum(A2)
sum(diag(A3))/sum(A3)

# split and train same as exercise 2 - easy to do it

## Exercise 5 - Benchmarking

## Exercise 6 - 

library(ISLR)
library(randomForest)

# ?Carseats
High = as.factor(ifelse(Carseats$Sales <= 8, 'No', 'Yes'))
CS = data.frame(Carseats, High)
CS$Sales = NULL

# grow a forest:
rf.out = randomForest(High~., CS)
# compare to bagging:
bag.out = randomForest(High~., CS, mtry=(ncol(CS)-1))

cbind(rf.out$importance, bag.out$importance)

par(mfrow=c(1,2))
varImpPlot(rf.out, pch=15, main="Ensemble method 1")
varImpPlot(bag.out, pch=15, main="Ensemble method 2")



### -- practice 2 --------------------------
set.seed(4061)
library(ISLR)
library(tree)
High = ifelse((Carseats$Sales<=8),'No','Yes')
CS = data.frame(Carseats, High)
CS$High = as.factor(CS$High)
CS$Sales = NULL

## class tree Q1
tr = tree(High~., data = CS)
summary(tr)
plot(tr)
text(tr, pretty = 0)
# prune it bro
pr_tr = cv.tree(tr, FUN = prune.misclass)
summary(pr_tr)
## use pruned tree best params
best_size = pr_tr$size[which.min(pr_tr$dev)]
new_pr = prune.misclass(tree = tr, best = best_size)
plot(new_pr)
text(new_pr, pretty = 0)


#/ part 2 ---/
set.seed(4061)
n = nrow(CS)
idxs = sample(1:n, n/2, replace = FALSE)
X = CS
Y = CS$High
X$High = NULL
train_X = X[idxs,]
train_Y = Y[idxs]
test_X = X[-idxs,]
test_Y = Y[-idxs]
tr_50 = tree(train_Y~.,train_X)
summary(tr_50)

tr_50_test = predict(tr_50, test_X, type = "class")
tr_50_test

pr_50 = cv.tree(tr_50, FUN = prune.misclass)
best_size_pr50 = pr_50$size[which.min(pr_50$dev)]
pr_50_final = prune.misclass(tr_50, best = best_size_pr50)
pr_50_preds = predict(new_pr, test_X, type = "class")

caret::confusionMatrix(tr_50_test, test_Y)
caret::confusionMatrix(pr_50_preds, test_Y)
# ROC
library(pROC)
vectors = predict(tr_50, test_X, type = "vector")
full_tree_roc = roc(response = test_Y, predictor = vectors[,1])
plot(full_tree_roc)
full_tree_roc$auc
vectors = predict(new_pr, test_X, type = "vector")
pr_tree_roc = roc(response = test_Y, predictor = vectors[,1])
plot(pr_tree_roc, add = TRUE, col = 4)
pr_tree_roc$auc
## part 3

## part 4
set.seed(4061)
n = nrow(CS)
idxs = sample(1:n, n/2, replace = FALSE)
X = CS
Y = CS$High
X$High = NULL
train_X = X[idxs,]
train_Y = Y[idxs]
test_X = X[-idxs,]
test_Y = Y[-idxs]
library(randomForest)

tr = tree(High~., data = CS)
summary(tr)
original_tree_preds = predict(tr, X, type="class")
c1 = table(Y, original_tree_preds)
c1
sum(diag(c1))/sum(c1)
rf_tree = randomForest(Y~., X)
rf_tree_preds = predict(rf_tree, X, type = "class")
c2 = table(Y, rf_tree_preds)
c2
# bagging
rf_tree_P = randomForest(Y~., X, mtry = ncol(X))
rf_tree_preds_P = predict(rf_tree_P, X, type = "class")
c3 = table(Y, rf_tree_preds_P)
c3

# 50-50
tr_50 = tree(train_Y~.,train_X)
summary(tr_50)
tr_50_preds = predict(tr_50, test_X, type = "class")
c4 = table(test_Y, tr_50_preds)
c4
sum(diag(c4))/sum(c4)

rf_tree_50 = randomForest(train_Y~., train_X)
rf_tree_preds_50 = predict(rf_tree_50, test_X, type = "class")
c5 = table(test_Y, rf_tree_preds_50)
c5
sum(diag(c5))/sum(c5)

rf_tree_bag = randomForest(train_Y~., train_X, mtry = ncol(X))
rf_tree_preds_bag = predict(rf_tree_bag, test_X, type = "class")
c6 = table(test_Y, rf_tree_preds_bag)
c6
sum(diag(c6))/sum(c6)

## Q6
library(ISLR)
library(randomForest)

# ?Carseats
High = as.factor(ifelse(Carseats$Sales <= 8, 'No', 'Yes'))
CS = data.frame(Carseats, High)
CS$Sales = NULL

# grow a forest:
rf.out = randomForest(High~., CS)
# compare to bagging:
bag.out = randomForest(High~., CS, mtry=(ncol(CS)-1))

cbind(rf.out$importance, bag.out$importance)

par(mfrow=c(1,2))
varImpPlot(rf.out, pch=15, main="Ensemble method 1")
varImpPlot(bag.out, pch=15, main="Ensemble method 2")

## Q7
library(gbm)
high = ifelse(Carseats$Sales <=8, 'No', 'Yes')
new_df = data.frame(Carseats, high)
set.seed(4061)
n = nrow(new_df)
X = new_df
y = new_df$high
y = (as.numeric(y =="Yes"))
X$Sales = NULL
X$high = NULL
idx = sample(1:n, 300)
X.train = X[idx,]
X.test = X[-idx,]
y.train = y[idx]
y.test = y[-idx]

gbm_tree= gbm(y.train~., X.train, 
              distribution="bernoulli",
              n.trees = 5000,
              interaction.depth = 1)

gbm_tree_preds = predict(gbm_tree, newdata=X.test, ntree = 5000)
caret::confusionMatrix(gbm_tree_preds, y.test)
roc.gb = roc(response=y.test, predictor=gbm_tree_preds)
plot(roc.gb)
roc.gb$auc
y.test



gbm_model = gbm(y.train~., data = X.train, 
                distribution = "bernoulli", n.trees = 5000,
                interaction.depth = 1)
gbm_model_preds = predict(gbm_model, newdata = X.test, 
                          ntree = 5000)

caret::confusionMatrix(gbm_model_preds, y.test)
roc.gb = roc(response=y.test, predictor=gbm_model_preds)
plot(roc.gb)
roc.gb$auc


## part 8
set.seed(4061)
library(caret)

dat = Hitters
dat = na.omit(dat)
dat$Salary = log(dat$Salary)

X = dat
Y = dat$Salary

n = nrow(X)
idxs = sample(1:n, round(0.7*n), replace = FALSE)
X.train = X[idxs,]
y.train = Y[idxs]
X.test = X[-idxs,]
y.test = Y[-idxs]
X.test$Salary = NULL

## dont use y.train~., X.train here as it's throwing error
caret_gbm = train(Salary~., X.train, method = 'gbm', distribution = 'gaussian')
gb.fitted = predict(caret_gbm)
gb.test.fitted = predict(caret_gbm, X.test)
gb.test.fitted
test_mse = mean((gb.test.fitted - y.test)^2)
test_mse
