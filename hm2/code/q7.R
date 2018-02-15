library(MASS)
library(glmnet)

train.x=as.matrix(read.csv("train.x",header=F))
train.y=as.matrix(read.csv("train.y",header=F))
test.x=as.matrix(read.csv("test.x",header=F))
test.y=as.matrix(read.csv("test.y",header=F))
model = glmnet(train.x, factor(train.y), family = "multinomial")
train.e = sum(as.numeric(predict(model, train.x, s = model$lambda[91], type = "class"))!=train.y)/length(train.y)
test.e = sum(as.numeric(predict(model, test.x, s = model$lambda[91], type = "class"))!=test.y)/length(test.y)
print(train.e)
print(test.e)

alphas = c(0, .2, .4, .6, .8, 1)
# 
pred.e = list()
data.dev = list()
for (i in 1:length(alphas)) {
  model = glmnet(train.x, factor(train.y), family = "multinomial", alpha = alphas[i])
  data.dev[[i]] = model$dev.ratio
  pred.e[[i]] = as.numeric(apply(predict(model, test.x, type = "class")!=as.vector(test.y), 2, mean))
  print(i)
}

m_colors = c("red4", "yellow1", "steelblue1", "springgreen2", "plum2", "pink1")
png(filename = "p7-plot.png")
plot(data.dev[[i]], pred.e[[i]], type="l", col=m_colors[1],
     lwd=2, xlab="% of deviance explained", ylab = "Test error")
title("% of deviance explained vs. Test error")
for (i in 2:length(alphas)) {
  lines(data.dev[[i]], pred.e[[i]], col=m_colors[i])
}
alpha_leg = character(length(alphas))
for (i in 1:length(alphas)) {
  alpha_leg[i] = paste("Alpha =", alphas[i])
}
legend(x = "topright", legend = alpha_leg, lwd = 5, col = m_colors)
