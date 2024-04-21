## Question 1
rm(list=ls())
require(gbm)
require(ISLR)
df = na.omit(Hitters)
df$Salary = log(df$Salary)
rates = c(0.001, 0.05, 0.01, 0.1)
L = length(rates)
set.seed(4061)

n = nrow(df)
B = 100

OOB_RMSEs = matrix(NA, nrow = B, ncol = L)
colnames(OOB_RMSEs) = rates

for (i in 1:B) {
  idxs = sample(1:n, n, replace=TRUE)
  X_train = df[idxs,]
  X_test = df[-idxs,]
  Y_test = df[-idxs,]$Salary
  for (j in 1:L){
    gbm_tree = gbm(Salary~., data=X_train, distribution = 'gaussian', shrinkage = rates[j])
    test_preds = predict(gbm_tree, X_test)
    OOB_RMSEs[i, j] = sqrt(mean((test_preds - Y_test)^2))
  }
}


for (i in 1:L) {
  for (j in 1:B){
    idxs = sample(1:n, n, replace=TRUE)
    X_train = df[idxs,]
    X_test = df[-idxs,]
    Y_test = df[-idxs,]$Salary
    gbm_tree = gbm(Salary~., data=X_train, distribution = 'gaussian', shrinkage = rates[i])
    test_preds = predict(gbm_tree, X_test)
    OOB_RMSEs[j, i] = sqrt(mean((test_preds - Y_test)^2))
  }
}
head(OOB_RMSEs)

## Question 1(1)
OOB_RMSEs_mean = apply(OOB_RMSEs, 2, mean)
round(OOB_RMSEs_mean, 4)

## Question 1(2)
par(mfrow=c(1,1))
boxplot(OOB_RMSEs, main="Boxplot of OOB-RMSEs across shrinkage rates",xlab="Shrinkage-values",ylab="Bootstrap RMSE-estimates", col = 'cyan') 

## Question 1(3)
# Reason - 

## Question 2
require(caret)
require(neuralnet)
require(DataExplorer)
df = read.csv(file="uws.csv", stringsAsFactors=TRUE)
subdf = df[,c("grade","sex","age","x.mean","x.max")]
y = df$grade
x = df
x$grade = NULL
str(df)

## Question 2(1)
## sex is a categorical feature

## Question 2(2)
plot_bar(df)

## Question 2(3)
plot_boxplot(subdf, by = 'grade')

## Question 2(4)
plot_boxplot(subdf, by = 'sex')

## Question 2(5)
age = subdf$age
sex = subdf$sex
x_max = subdf$x.max
x_mean = subdf$x.mean

wilcox.test(age ~ sex, alternative = "two.sided")
wilcox.test(x_max ~ sex, alternative = "two.sided")
wilcox.test(x_mean ~ sex, alternative = "two.sided")

## Question 2(6)
conversion <- function(x){
  # function recoding levels into numerical values
  if(is.factor(x)){
    levels(x)
    return(as.numeric(x)) 
  } else {
    return(x)
  }
}
scaling <- function(x){
  # function applying normalization to [0,1] scale
  mins = min(x,na.rm=TRUE)
  maxs = max(x,na.rm=TRUE)
  return((x-mins)/(maxs-mins))
}
df_inter = data.frame(lapply(df,conversion))
df_scaled = data.frame(lapply(df_inter,scaling))
means = apply(df_scaled[, c("age", "x.mean", "x.max")], 2, mean)
round(means, 4)
df_scaled$grade = NULL

## Question 2(7)
set.seed(4061)
mod = neuralnet(y~., data = df_scaled, hidden=c(5), linear.output = FALSE)
(error = mod$result.matrix["error",])

## Question 2(8)
col_names = colnames(mod$response)
final_preds = as.factor(col_names[max.col(predict(mod, df_scaled))])
cf_mat = caret::confusionMatrix(final_preds, y)
(overall_accuracy = cf_mat$overall[1])
specificity_class = cbind(cf_mat$byClass[1],cf_mat$byClass[2],cf_mat$byClass[3])
colnames(specificity_class) = c("high","int","low")
round(specificity_class, 4)

## Question 2(9)
x$sex = as.numeric(x$sex)
correlation_matrix <- cor(x)
cols = colnames(x)
pairs = c()
for (i in cols) {
  for (j in cols) {
    if (i!=j & abs(correlation_matrix[i,j]) > 0.95) {
      pairs = c(pairs, c(i,j))
    }
  }
}
cat("Columns to remove : ", unique(pairs))
unique(pairs)
