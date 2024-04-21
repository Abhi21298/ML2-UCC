# --------------------------------------------------------
# ST4061 / ST6041
# 2023-24
# Eric Wolsztynski
# ...
# Exercises Section 2: Regularization
# --------------------------------------------------------

###############################################################
### Exercise 1: tuning LASSO
###############################################################

# Have a go at this exercise yourself...
# you can refer to ST4060 material:)

library(ISLR)
library(glmnet)
# ?glmnet

dat = na.omit(Hitters)
n = nrow(dat)
set.seed(4061)
dat = dat[sample(1:n, n, replace=FALSE),]
dat$Salary = log(dat$Salary)
x = model.matrix(Salary~., data=dat)[,-1]
y = dat$Salary
lambda_grid = 10 ^ seq(10,-2, length.out= 100)

lasso.m = glmnet(x,y, lambda = lambda_grid)
ridge.m = glmnet(x,y, lambda = lambda_grid, alpha = 0)
lasso.m
coef(lasso.m, 0.1) # change lambda values to get resp coefficients!
# fix this by using cv.glmnet to get optimum lambda value
?cv.glmnet
?glmnet

############################################################### 
### Exercise 2: tuning LASSO + validation split
############################################################### 

# Have a go at this exercise yourself too...
# you can refer to ST4060 material:)
set.seed(1)
train_ind = sample(c(TRUE,FALSE), n, replace= TRUE, prob = c(0.7,0.3))
x = model.matrix(Salary~., data=dat[train_ind,])[,-1]
y = dat$Salary[train_ind]
lambda_lasso = cv.glmnet(x, y)
lambda_ridge = cv.glmnet(x, y, alpha = 0)
lasso_model = glmnet(x, y, lambda = lambda_lasso$lambda.min)
ridge_model = glmnet(x, y, lambda = lambda_ridge$lambda.min, alpha = 0)
x_test = model.matrix(Salary~., data=dat[-train_ind,])[,-1]
lasso_preds = predict(lasso_model, newx = x_test)
ridge_preds = predict(ridge_model, newx = x_test)
colnames(lasso_preds) = c('lasso')
y_test = dat$Salary[-train_ind]
rmse_lasso = sqrt(sum((y_test - lasso_preds) ^ 2))
rmse_ridge = sqrt(sum((y_test - ridge_preds) ^ 2))
rmse_lasso
rmse_ridge
cbind(y_test, lasso_preds, ridge_preds)
