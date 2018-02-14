xtrain=as.matrix(read.csv("train.x",header=F))
ytrain=as.matrix(read.csv("train.y",header=F))
xtest=as.matrix(read.csv("test.x",header=F))
ytest=as.matrix(read.csv("test.y",header=F))

library(glmnet)
l=glmnet(xtrain,factor(ytrain),family="multinomial")
e1=sum(as.numeric(predict(l,xtrain,s=l$lambda[99],type="class"))!=
ytrain)/length(ytrain)
e2=sum(as.numeric(predict(l,xtest,s=l$lambda[99],type="class"))!=
ytest)/length(ytest)
print(e1)
print(e2)
