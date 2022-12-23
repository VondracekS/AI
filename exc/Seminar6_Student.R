# rm(list=ls())
zsnew = read.csv(file='~//zsnew.csv')
summary(zsnew$default)

set.seed(200)
idx = sample(1:dim(zsnew)[1],size=floor(dim(zsnew)[1]*0.8),replace=F)
train = zsnew[+idx,]
test  = zsnew[-idx,]

# Create alternative datasets - that are more balanced
idx.good = which(train$default==0); length(idx.good)
idx.bad  = which(train$default==1); length(idx.bad)

# Undersampling majority
set.seed(300)
tmp = idx.good[sample(1:length(idx.good),size=length(idx.bad),replace=F)]
train.um = rbind(train[tmp,],train[idx.bad,])
dim(train.um)
summary(train.um$default)
# Just some randomization - we do not want defaults and non-deafults to be stacked one next to each other (just to be sure)
train.um = train.um[sample(1:dim(train.um)[1],size=dim(train.um)[1],replace=F),]

# Undersampling majority and oversampling minority (2 times the number of minority)
set.seed(300)
tmp.good = idx.good[sample(1:length(idx.good),size=length(idx.bad)*2,replace=F)]
# Here we need replacement = T - why?
tmp.bad  = idx.bad[sample(1:length(idx.bad),size=length(idx.bad)*2,replace=T)] # Oversampling
train.uo = rbind(train[tmp.good,],train[tmp.bad,])
# Just some randomization - we do not want defaults and non-deafults to be stacked one next to each other (just to be sure)
train.uo = train.uo[sample(1:dim(train.uo)[1],size=dim(train.uo)[1],replace=F),]
dim(train.uo)
summary(train.uo$default)

# Cost weighted
wgt   = rep(0,dim(train)[1])
wgt[train$default==1] = 0.5/length(idx.bad)
wgt[train$default==0] = 0.5/length(idx.good)
train$wgt = wgt

# Let's estimate couple of models
spec_one = as.formula(default~amount+term+rate+MIRR2+time.start+time.start2)

# PLM
p1 = predict(lm(spec_one,data=train),new=test)
p2 = predict(lm(spec_one,data=train.um),new=test)
p3 = predict(lm(spec_one,data=train.uo),new=test)
p4 = predict(lm(spec_one,data=train,weights=wgt),new=test)

# We make some adjustments - why?
p1[p1>1] = 0.999
p1[p1<0] = 0.001
p2[p2>1] = 0.999
p2[p2<0] = 0.001
p3[p3>1] = 0.999
p3[p3<0] = 0.001
p4[p4>1] = 0.999
p4[p4<0] = 0.001

# LR
p5 = predict(glm(spec_one,data=train,family='binomial'),new=test)
p5 = 1/(1+exp(-p5))
p6 = predict(glm(spec_one,data=train.um,family='binomial'),new=test)
p6 = 1/(1+exp(-p6))
p7 = predict(glm(spec_one,data=train.uo,family='binomial'),new=test)
p7 = 1/(1+exp(-p7))
p8 = predict(glm(spec_one,data=train,weights=wgt,family='binomial'),new=test)
p8 = 1/(1+exp(-p8))

# We will make some adjustments even here - why? Completly different reason - cross-entropy
p5[p5>0.999] = 0.999
p5[p5<0.001] = 0.001
p6[p6>0.999] = 0.999
p6[p6<0.001] = 0.001
p7[p7>0.999] = 0.999
p7[p7<0.001] = 0.001
p8[p8>0.999] = 0.999
p8[p8<0.001] = 0.001

# Predictions
predicts = cbind(test$default,p1,p2,p3,p4,p5,p6,p7,p8)

# Data-frame with results: Accuracy, Specificity, Sensitivity, Precision, AUC, BS, CE
library(caret)
library(MCS)
library(pROC)

results = data.frame(metrics = c('Accuracy','Specificity','Sensitivity','Precision','AUC','Brier score','Cross entropy'),
                     p1=NA,p2=NA,p3=NA,p4=NA,p5=NA,p6=NA,p7=NA,p8=NA)
reference = as.factor(predicts[,1])
for (i in 2:9) {
  prediction = as.factor((predicts[,i]>0.5)*1)
  tmp = confusionMatrix(data=prediction,reference=reference,positive='1')
  
  # Brier score
  brier.loss = (predicts[,i] - test$default)^2
  
  # Cross entropy
  # PLM
  cross.loss = -1*(log(predicts[,i])*test$default + log(1-predicts[,i])*(1-test$default))
  
  results[,i] = c(tmp$overall[1], tmp$byClass[c(2,1,5)],
                  as.numeric(roc(response=predicts[,1],predictor=predicts[,i])$auc),
                  mean(brier.loss),mean(cross.loss))
}


# Let's try to improve the zsnew models

# Try the same with the TITANIC
titanic = read.csv(file='~\\titanic.csv')
