rm(list = ls())
require(ISLR)
require(class)
require(pROC)
library(randomForest)

x = Smarket[,-9]
y = Smarket$Direction
set.seed(4061)
train = sample(1:nrow(Smarket),1000)

## Question 3 i)

rf.tree = randomForest(y[train] ~ ., data = x[train,])
rf.tree
#summary(rf.tree)
rf.tree.preds = predict(rf.tree, x[train,], type = 'class')
prediction.conf = table(rf.tree.preds, y[train])
prediction.conf
missclass_rate = (1 - (sum(diag(prediction.conf))/sum(prediction.conf)))
missclass_rate

## Question 3 ii)
y_test_true = y[-train]
test_preds = predict(rf.tree, newdata=x[-train,], type='class')
#(rf.test.confusion = table(test_preds, y_test_true))
rftree.probs = predict(rf.tree, x[-train,], type="prob")
roc = roc(response=y_test_true, predictor=rftree.probs[,2])
auc = roc$auc
auc
plot(roc, col=1)

## Question 3 iii)
k = 2
knn.o = knn(x[train,], x[-train,], y[train], k)
knn.preds = as.numeric(knn.o == 'Up')
knn.p = attributes(knn(x[train,], x[-train,], y[train], k, prob=TRUE))$prob
new.probs = 1 - knn.p
final.knn.preds = ifelse(knn.preds == 1,knn.p, new.probs)
roc_knn = roc(y_test_true, final.knn.preds)
plot(roc_knn, add = TRUE, col = 75)
legend("bottomright", legend = c("Random Forest", "KNN : k=2"), col = c(1, 75), lty = 1, lwd = 2)
auc_knn = roc_knn$auc
auc_knn

## Question 3 iv)
set.seed(4061)
M = 1000
train = sample(1:nrow(Smarket), M)

K = 10
test_class_errors = numeric(K)*NA
for(k in 1:K) {
  knn.o = knn(x[train,], x[-train,], y[train], k)
  confusion_mat = table(knn.o, y[-train])
  test_class_errors[k] = (1 - (sum(diag(confusion_mat))/sum(confusion_mat)))
}
test_class_errors
plot(seq(1:K), test_class_errors, xlim = c(1,10),
     xlab = "k-values", ylab = "Test-Misclassification Error rate", 
     main = paste("Test-set Misclassification errors for KNN with k-values = 1:",K,sep=''), 
     col = 4, type = 'l')
points(seq(1:K), test_class_errors, col=1, pch=20, cex = 1.4)
axis(side = 1, at = seq(1, 10, by = 1), labels = seq(1, 10, by = 1))
