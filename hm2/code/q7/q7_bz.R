library(MASS)

train.3 = read.csv("train.3", header=F)
train.5 = read.csv("train.5", header=F)
train.8 = read.csv("train.8", header=F)


# data processing
xtrain = rbind(as.matrix(train.3), as.matrix(train.5), 
               as.matrix(train.8))
ytrain = rep(c(3,5,8), c(nrow(train.3), nrow(train.5), nrow(train.8)))
test = as.matrix(read.table("zip.test"))
ytest = test[,1]
xtest = test[ytest==3 | ytest == 5 | ytest == 8, -1]
ytest = ytest[ytest == 3 | ytest == 5 | ytest == 8]

showDigit <- function(x, ...) {
  p = length(x)
  d = sqrt(p)
  if (d != round(d)) stop("must be perfect square")
  x = matrix(as.numeric(x), d,d)
  image(x[,d:1], axes=F, ...)
}

# output
pred_errors = matrix(0, nrow=5, ncol=2)

# part a
l = lda(xtrain, ytrain)
pred_errors[1,1] = sum(predict(l)$class != ytrain) / length(ytrain)
pred_errors[1,2] = sum(predict(l, xtest)$class != ytest) / length(ytest)
# print(pred_errors)

# part b
xtrain.center <- apply(xtrain, 2, mean)
xxtrain = scale(xtrain, center = xtrain.center, scale = F)
xxtest = scale(xtest, center = xtrain.center, scale = F)
V = svd(xxtrain)$v[,1:30]
pcstrain = xxtrain %*% V
pcstest = xxtest %*% V

l = lda(pcstrain, ytrain)
pred_errors[2,1] = sum(predict(l)$class != ytrain) / length(ytrain)
pred_errors[2,2] = sum(predict(l, pcstest)$class != ytest) / length(ytest)

# part c
V = svd(xxtrain)$v[, 1:10]
pcstrain = xxtrain %*% V
pcstest = xxtest %*% V

l = lda(pcstrain, ytrain)
pred_errors[3,1] = sum(predict(l)$class != ytrain) / length(ytrain)
pred_errors[3,2] = sum(predict(l, pcstest)$class != ytest) / length(ytest)

# part d
filterdigit <- function(x) {
  # average each non-overlapping 2x2 block
  x = matrix(x, 16,16)
  twos = rep(1:4, 4)
  x = x[twos == 1,] + x[twos==2,] + x[twos == 3,] + x[twos==4,]
  x = x[,twos == 1] + x[,twos==2] + x[,twos == 3] + x[,twos==4]
  as.vector(x)/16
}
xtrain = t(apply(xtrain, 1, filterdigit))
xtest = t(apply(xtest, 1, filterdigit))
l = lda(xtrain, ytrain)
pred_errors[4,1] = sum(predict(l)$class != ytrain) / length(ytrain)
pred_errors[4,2] = sum(predict(l, xtest)$class != ytest) / length(ytest)

# part e
library(glmnet)
#print("here1")
l = glmnet(xtrain, factor(ytrain), family = "multinomial")
#print(l)
#print("here2")
pred_errors[5,1] = sum(as.numeric(predict(
                        l, xtrain, s=l$lambda[90], type = "class"))
                       != ytrain) / length(ytrain)
pred_errors[5,2] = sum(as.numeric(predict(
                        l, xtest, s = l$lambda[90], type = "class"))
                       != ytest) / length(ytest)
# print(pred_errors)

# output prediction scores
print(round(pred_errors, 4))


# plot of deviance explained vs. test err
alpha.values = c(0, .25, .5, .75, 1)
# 
pred.err = list()
dev.explained = list()
for (i in 1:length(alpha.values)) {
  l = glmnet(xtrain, factor(ytrain), family = "multinomial", 
             alpha = alpha.values[i])
  dev.explained[[i]] = l$dev.ratio
  pred.err[[i]] = as.numeric(apply(
                    predict(l, xtest, type = "class") != ytest, 2, mean))
}

my.colors = c("red4", "yellow1", "steelblue1", "springgreen2", "plum2", "pink1")
png(filename = "p7-plot.png")
plot(dev.explained[[i]], pred.err[[i]], type="l", col=my.colors[1],
     lwd=2, xlab="% of deviance explained", ylab = "Test error")
title("% of deviance explained vs. Test error")

for (i in 1:length(alpha.values)) {
  lines(dev.explained[[i]], pred.err[[i]], col=my.colors[i], lwd=2)
}

alpha.legend = character(length(alpha.values))
for (i in 1:length(alpha.values)) {
  alpha.legend[i] = paste("Alpha =", alpha.values[i])
}
legend(x = "topright", legend = alpha.legend, lwd = 5, col = my.colors)
dev.off()


