## section 6 - Neural networks!

## Q1
library(MASS)
library(neuralnet)
nms = names(Boston)[-14]
set.seed(4061)
f = as.formula(paste("medv ~", paste(nms, collapse = " + ")))
out.nn = neuralnet(f, data=Boston, hidden=c(10), rep=5, linear.output=FALSE)

## feed forward neural network
Boston
summary(out.nn)
out.nn$response ## do not use this for predicting!!
out.nn$act.fct
out.nn.lin = neuralnet(f, data=Boston, hidden = c(10), rep = 5, linear.output = TRUE)
## did not coverge properly within the given set. too long to train.

## create tanh activation function!,
## Use linear.output = FALSE for applying act function!
out.nn.tanh = neuralnet(f, data=Boston, hidden=c(10), rep=5, act.fct = 'tanh', linear.output=FALSE)

## always use predict function for neural networks!!
preds.logistic.act = predict(out.nn, Boston) ## correct way
preds.tanh.act = predict(out.nn.tanh, Boston) 
sqrt(mean((preds.logistic.act - Boston$medv)^2))
sqrt(mean((preds.tanh.act - Boston$medv)^2))


## Question 2 - Check if outcome is factor! 
set.seed(4061)
df = iris[sample(1:nrow(iris)),]
cols = names(iris[,-5])
cols
f = as.formula(paste("Species ~", paste(cols, collapse = " + ")))
f
iris.nn = neuralnet(f, data=df, hidden=c(6,5))
pds = predict(iris.nn, iris)
colnames(pds) = levels(iris$Species)
pds
max.col(pds)
plot(iris.nn)


## Question 3
library(neuralnet)
library(nnet)    # implements single layer NNs
library(MASS) # includes dataset BostonHousing
set.seed(4061)
n = nrow(Boston)
itrain = sample(1:n, round(.7*n), replace=FALSE)
dat = Boston
dat$medv = dat$medv/50
dat.train = dat[itrain,]
dat.test = dat[-itrain,-14]
y.test = dat[-itrain,14]


lib.nnet.mod = nnet(medv~.,data=dat.train, size = 5, linout = TRUE)
lib.nnet.mod$fitted.values ## for nnet you can use fitted vals
## giving same values
predict(lib.nnet.mod, dat.train) ## same as above for nnet only

## dont fitted.values for neuralnet library trained model
nnet2 = neuralnet(medv~., data=dat.train, hidden = c(5), linear.output =  TRUE)
fit2 = predict(nnet2, dat.train)[,1]
mean((fit2 - dat.train$medv)^2)

## threshold for partial derivates of weights we use! 
set.seed(4061)
nms = names(dat)[-14]
f = as.formula(paste("medv ~", paste(nms, collapse = " + ")))
nno3 = neuralnet(f, data=dat.train, hidden=5, threshold=0.0001)
fit3 = predict(nno3, newdata=dat.train)[,1]
mean((fit3-dat.train$medv)^2)


## QUestion 4 - regression nnet
dat = na.omit(Hitters)
set.seed(4061)
n = nrow(dat)
itrain = sample(1:n, round(.7*n), replace=FALSE)
dat.train = dat[itrain,]

nnet.regr = nnet(Salary~., data=dat, size = 6, linout = TRUE)
preds = nnet.regr$fitted.values[,1]
preds
mean((preds-dat.train$Salary)^2)


## use decay for regularization - decay=c(0) = default!
set.seed(4061)
nnet.regr.dec = nnet(Salary~., data=dat, size = 6, linout = TRUE, decay = c(0.1))
preds2 = nnet.regr.dec$fitted.values[,1]
preds2
mean((preds2-dat.train$Salary)^2)

## sir's code - he is doing scaling! 
library(caret)
library(neuralnet)
library(nnet)
library(ISLR)

# set up the data (take a subset of the Hitters dataset)
dat = na.omit(Hitters) 
n = nrow(dat)
NC = ncol(dat)

# Then try again after normalizing the response variable to [0,1]:
dats = dat
dats$Salary = (dat$Salary-min(dat$Salary)) / diff(range(dat$Salary))

# train neural net
set.seed(4061)
itrain = sample(1:n, round(.7*n), replace=FALSE)
dat.train = dat[itrain,]
dats.train = dats[itrain,]
dat.test = dat[-itrain,]
dats.test = dats[-itrain,]

set.seed(4061)
nno = nnet(Salary~., data=dat.train, size=10, decay=c(0.1))
summary(nno$fitted.values)

set.seed(4061)
nno.s = nnet(Salary~., data=dats.train, size=10, decay=c(0))
summary(nno.s$fitted.values)

set.seed(4061)
nno.s = nnet(Salary~., data=dats.train, size=10, decay=c(0.1))
summary(nno.s$fitted.values)

# Our last attempt above was a success.
# But we should be able to get a proper fit even for decay=0... 
# what's going on? Can you get it to work?

# (A1) Well, it's one of these small details in how you call a function;
# here we have to specify 'linout=1' because we are considering a 
# regression problem:

set.seed(4061)
nno = nnet(Salary~., data=dat.train, size=10, decay=c(0.1), linout=1)
summary(nno$fitted.values)

set.seed(4061)
nno.s = nnet(Salary~., data=dats.train, size=10, decay=c(0), linout=1)
summary(nno.s$fitted.values)

set.seed(4061)
nno.s = nnet(Salary~., data=dats.train, size=10, decay=c(0.1), linout=1)
summary(nno.s$fitted.values)

# (A2) but let's do the whole thing again more cleanly...

# re-encode and scale dataset properly
myrecode <- function(x){
  # function recoding levels into numerical values
  if(is.factor(x)){
    levels(x)
    return(as.numeric(x)) 
  } else {
    return(x)
  }
}
myscale <- function(x){
  # function applying normalization to [0,1] scale
  minx = min(x,na.rm=TRUE)
  maxx = max(x,na.rm=TRUE)
  return((x-minx)/(maxx-minx))
}
datss = data.frame(lapply(dat,myrecode))
datss = data.frame(lapply(datss,myscale))

# replicate same train-test split:
datss.train = datss[itrain,]
datss.test = datss[-itrain,]

set.seed(4061)
nno.ss.check = nnet(Salary~., data=datss.train, size=10, decay=0, linout=1)
summary(nno.ss.check$fitted.values)

# use same scaled data but with decay as before:
set.seed(4061)
nno.ss = nnet(Salary~., data=datss.train, size=10, decay=c(0.1), linout=1)
summary(nno.ss$fitted.values)

# evaluate on test data (with same decay for both models):
datss.test$Salary - dats.test$Salary
pred.s = predict(nno.s, newdata=dats.test)
pred.ss = predict(nno.ss, newdata=datss.test)
mean((dats.test$Salary-pred.s)^2)
mean((datss.test$Salary-pred.ss)^2)


### Question 5 - caret nnet and mlpML
datss.train # scaled data
names(datss.train)

nnet5 = nnet(Salary~.,data=datss.train, size = 10)
ncol(datss.train)
x = datss.train[,-19]
y = datss.train[,19]
names(x)
y
m = train(x, y, method ="nnet", tuneGrid = expand.grid(decay = c(0.1), size = c(5)), linout = 1)

## Q7

#m2 = train(x, y, method ="mlpML", tuneGrid = expand.grid(decay = c(0.1), layer1 = c()), linout = 1)
