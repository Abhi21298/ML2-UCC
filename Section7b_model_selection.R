# --------------------------------------------------------
# ST4061 / ST6041
# 2023-24
# Eric Wolsztynski
# ...
# Exercises Section 7: model selection
# * Best subset selection
# * Forward/Backward subset selection
# This scripts demos both the regsubsets() and step() 
# functions for stepwise selection based on linear
# modelling.
# --------------------------------------------------------

library(ISLR)
library(leaps) # contains regsubsets()

###############################################################
### Exercise 1: best subset selection
###############################################################

rm(list=ls())

Hitters = na.omit(Hitters)
dim(Hitters)

# (1) best subset selection

set.seed(4061)
reg.full = regsubsets(Salary~., data=Hitters, method="exhaustive")
# cf. ?regsubsets: the function is written with argument
# method=c("exhaustive","backward", "forward", "seqrep"),
# i.e. method="exhaustive" by default (the 1st option in the list), 
# although it's better practice to specify it anyway...
reg.full

# (2) inspect output
# ?regsubsets

names(summary(reg.full))
summary(reg.full)
summary(reg.full)$which      # tracks covariate selection
summary(reg.full)$outmat

# (3) plot RSS v p

RSS = summary(reg.full)$rss
plot(RSS, pch=20, t='b', xlab="Number of covariates", ylab='RSS')
# see also...
R2adj = summary(reg.full)$adjr2
plot(R2adj, pch=20, t='b', xlab="Number of covariates", 
	ylab='Adjusted R^2')

# (4) increase model size up to 19 and identify RSS-optimal
#     and Adjusted R^2-optimal models

set.seed(4061)
reg.full = regsubsets(Salary~., data=Hitters, nvmax=19)

par(mfrow=c(1,2))
RSS = summary(reg.full)$rss
plot(RSS, pch=20, t='b', xlab="Number of covariates", ylab='RSS')
R2 = summary(reg.full)$rsq
R2adj = summary(reg.full)$adjr2
plot(R2, pch=20, t='b', xlab="Number of covariates", 
	ylab='Original and adjusted R^2')
points(R2adj, col=4, pch=15, t='b')

# Keep track of optima:
R2adj.index = which.max(R2adj)
# visualize:
abline(v = R2adj.index)
# extract find corresponding model:
summary(reg.full)$outmat[R2adj.index,]
mod = summary(reg.full)$which[R2adj.index,]
names(which(mod))

# feature selection heatmap:
dev.new()
plot(reg.full, scale="adjr2")
plot(reg.full, scale="bic")
summary(reg.full)$outmat

# can you track this matrix?
# ?plot.regsubsets

###############################################################
### Exercise 2: fwd/bwd subset selection
###############################################################

rm(list=ls())

Hitters = na.omit(Hitters)
dim(Hitters)

set.seed(4061)
reg.fwd = regsubsets(Salary~., data=Hitters, nvmax=19, method="forward")
set.seed(4061)
reg.bwd = regsubsets(Salary~., data=Hitters, nvmax=19, method="backward")

par(mfrow=c(1,2))
#
plot(summary(reg.bwd)$rss, t='b', pch=20, cex=1.5)
points(summary(reg.fwd)$rss, t='b', pch=15, col=2)
#
plot(summary(reg.bwd)$adjr2, t='b', pch=20, cex=1.5)
points(summary(reg.fwd)$adjr2, t='b', pch=15, col=2)

R2adj.fwd.index = which.max(summary(reg.fwd)$adjr2)
R2adj.bwd.index = which.max(summary(reg.bwd)$adjr2)
R2adj.bwd.index
abline(v=c(R2adj.bwd.index, R2adj.fwd.index), col=c(1,2))

# 4-variable models:
coef(reg.fwd, id=4)
coef(reg.bwd, id=4)
# compare with "best subset" approach:
reg.full = regsubsets(Salary~., data=Hitters, method="exhaustive")
names(which(summary(reg.full)$which[4,]==TRUE))

names(which(summary(reg.fwd)$which[11,]==TRUE))
names(which(summary(reg.bwd)$which[11,]==TRUE))
# extract optimal model from backward elimination process:
coef(reg.bwd, id=R2adj.bwd.index)

###############################################################
### Exercise 3: generate predictions from regsubsets() output?
###############################################################

# How to generate predictions from regsubsets() output?

rm(list=ls())

set.seed(4061)
dat = na.omit(Hitters)

n = nrow(dat)
itrain = sample(1:n, 150)
reg.fwd = regsubsets(Salary~., data=dat, nvmax=10, 
						method="forward", subset=itrain)

# now predict?
predict(reg.fwd, newdata=dat[-itrain,])
# Aaargh... there's no predict() function for regsubsets!!
# So we must compute the predicted values manually.
# Here we need to reconstruct the 4-variable model fit...

# ... we can evaluate the linear model equation directly
# get vector of estimates:
beta.hat = coef(reg.fwd, id=4)    
# create matrix X:
test.dat = model.matrix(Salary~., data = dat[-itrain,])
# get the test data matrix:
Xtest = test.dat[,names(beta.hat)]
# compute (X Beta^T) as in Y = (X Beta^t) + Epsilon:
pred = Xtest %*% beta.hat  
pred = as.numeric(pred)     # make this a vector instead
# compute prediction RMSE:
sqrt( mean((pred - dat$Salary[-itrain])^2) )

###############################################################
### Exercise 4: fit and predict using stats::step()
###############################################################

rm(list=ls())

Hitters = na.omit(Hitters)
dim(Hitters)

# stepwise  selections from regsubsets():
set.seed(4061)
reg.fwd = regsubsets(Salary~., data=Hitters, nvmax=19, method="forward")
set.seed(4061)
reg.bwd = regsubsets(Salary~., data=Hitters, nvmax=19, method="backward")

# stepwise  selections from step():
lm.out = lm(Salary~., data=Hitters) # full model
lm.start = lm(Salary~1, dat=Hitters) # null model
set.seed(4061)
step.bth = step(lm.out, direction="both")
set.seed(4061)
step.fwd = step(lm.out, direction="forward")

## for full model use the null model and give scope, 
## you will arrive at the same result as that of backward!
set.seed(4061)
step.fwd = step(lm.start, direction="forward", scope=formula(lm.out))

set.seed(4061)
step.bck = step(lm.out, direction="backward")

# compare backward selections from regsubsets() and step():
#
# ... from step()...
coef(step.bck)
length(coef(step.bck))
summary(step.bck) 
# Nice: we get the model directly! No need to reconstruct 
# fitted values by hand this time!
#
# ... from regsubsets()...
i.opt = which.min(summary(reg.bwd)$bic)
i.set.opt = summary(reg.bwd)$which[i.opt,]
summary(reg.bwd)$which[i.opt, i.set.opt]

# Different models, but smaller one is included in larger one.
# Difference is likely due to using different criteria 
# (AIC v BIC, BIC yielding a smaller model).

# NB: we can also assess feature contributions in terms 
# of magnitude of their effect:
coefs = abs(coef(step.fwd))/sum(abs(coef(step.fwd)), na.rm=TRUE)*100
coefs = coefs[-1]
coefs[order(abs(coefs), decreasing=TRUE)]

###############################################################
### Exercise 5: generate predictions from step() output?
###############################################################

rm(list=ls())

set.seed(4061)
dat = na.omit(Hitters)

n = nrow(dat)
itrain = sample(1:n, 150)
lmo = lm(Salary~., data=dat, subset=itrain)
reg.fwd = step(lmo, direction="forward")

# now predict?
pred = predict(reg.fwd, newdata=dat[-itrain,])
sqrt(mean((pred - dat$Salary[-itrain])^2))

# much easier than regsubsets() on this front...

