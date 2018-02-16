setwd("/home/charlie/Documents/Challenge/")
rm(list = ls())

## LOADING DATA
load("Rda/train_18_all.Rda")
load("Rda/train_output_18_all.Rda")

# load("tr_signif_9122.Rda")
load("Rda/p_values.Rda")
# load("Rda/test_18.Rda")
load("Rda/val_18_all.Rda")
load("Rda/val_output_18_all.Rda")

N  = nrow(tr)
P  = ncol(tr)

## COMPUTE PVALUES IF NOT ALREADY DONE
if(exists("tr_signif")){
  print("significants SNPs found ...")
  sig_SNP = colnames(tr_signif)
}


if(!exists("tr_signif")){
  print("Finding significants SNPs using LR tests ...")
  p_values_df = data.frame(colnames(tr),rep(0,P))
  colnames(p_values_df) <- c("SNP","pval")
  
  
  for(i in 1:P){
    model_logistic = glm(tr_out~tr[,i],family = "binomial") ## ful mod
    p_val = summary(model_logistic)$coefficients[2,4] # p-value of LR test
    p_values_df$pval[i] = p_val
    if(i%%1000 ==0) print(i) # know where we are
  }
  
  #order p values
  ord = order(p_values_df$pval)
  p_values_df$SNP  = p_values_df$SNP[ord]
  p_values_df$pval = p_values_df$pval[ord]
  
  head(p_values_df)
  
  # do bonferonni adjusting ()
  Bonferroni <-  p.adjust(p_values_df$pval,method="bonferroni")
  print(sum(Bonferroni<0.05))
  
  ## manual way of doing Bonferonni
  # sum(p_values_df$pval<((10^-5)/P))
  
  
  # Compute significant SNP based on Bonferroni
  sig_SNP = p_values_df$SNP[which( (Bonferroni<(0.05)) !=0)]
  
  # save(tr_signif,file="tr_signif_2000+.Rda") # IF YOU WANT TO SAVE SIGNIFICANT SNPs
}


##### CRAMER TO DETECT REDUNDANT FEATURES #####
# library("DescTools")
# 
# for(i in 1:2744){
#   print(i)
#   if(tr_signif[1,i]!=-1){# i.e not already removed
#     for(j in (i+1):2744){
#       v = CramerV(tr_signif[,j],tr_signif[,i])
#       if(v>0.8 && tr_signif[1,j]!=-1){ # i.e not already removed and significantly associated
#         print(v)
#         tr_signif[1,j] = -1
#       }
#     }
#   }
# }


######## create dataframe ##########
tr_signif = tr[,sig_SNP] # matrix of significant SNPs IF NEEDED
# save(tr_signif,file="tr_signif_3066.Rda") # IF YOU WANT TO SAVE SIGNIFICANT SNPs

val=val[,sig_SNP] # keep only significant features

df = data.frame(tr_signif,tr_out) # avoid predict bugs of length

# PCA #######



########  RANDOM FOREST #########
library(caret)
library(xgboost)
df = data.frame(tr_signif_LASSO,tr_out)
# tr <- trainControl(method = "cv", number = 5)
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        #summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel=T)

xgb.grid <- expand.grid(nrounds = 1000,
                        eta = c(0.01,0.05,0.1),
                        max_depth = c(2,4,6,8,10,14)
)
set.seed(45)
xgb_tune <-train(as.factor(tr_out)~as.matrix(tr_signif_LASSO),
                 data=df,
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="Kappa",
                 nthread =3
)


  



library(randomForest)
# tr_out <- as.factor(tr_out) # If a factor, classification is assumed, otherwise regression is
                            # assumed. If omitted, randomForest will run in unsupervised mode.
set.seed(1)              
res <- tuneRF(x = tr_signif_LASSO,
              y = as.factor(tr_out),
              ntreeTry = 1000,mtryStart = 10)

# Look at results
print(res)

# Find the mtry value that minimizes OOB Error
mtry_opt <- res[,"mtry"][which.min(res[,"OOBError"])]
print(mtry_opt)

# model = randomForest(tr_signif_LASSO,as.factor(tr_out),ntree = 1000,do.trace =10,sampsize=c(10000,10000),mtry=23)


pred_tr <- predict(model,tr_signif,type="prob")
pred_val <- predict(model,val,type="prob")
pred_tr <- pred_tr[,2]
pred_val <- pred_val[,2]


pred_te <- predict(model,test,type="prob")
pred_te = pred_te[,2]


########  SVM #########
library("e1071")
res = svm(tr_signif,tr_out,probability = TRUE)
pred_tr <- fitted(res)
sum(pred_tr!= tr_out)/N


pred_val <- predict(res,val)
sum(pred_val!= tr_out_val)/length(tr_out_val)


########  Logistic Regression #########
# SHOULD I SCALE FEATURE MATRIX EVEN THOUGH IT'S CATEGORICAL ?? 
# does not seem to impact result


model = glm(tr_out ~ .,data=df,family="binomial")

pred_tr = predict(model,tr_signif,type="response")

pred_val = predict(model,newdata = val,type="response")

test = test[,sig_SNP]
pred_te <- predict(model,newdata = test,type="response")

#### LASSO #####
library(glmnet)
model <- glmnet(as.matrix(tr_signif),as.factor(tr_out),alpha=1,family="binomial")



cv.model <- cv.glmnet(as.matrix(tr_signif), y=tr_out, alpha=1)
best.lambda <- cv.model$lambda.min
# best lambda is often 0.00083
pred_tr = predict(model,s=best.lambda,as.matrix(tr_signif),type="response")
pred_val = predict(model,s=best.lambda,as.matrix(val),type="response")
pred_te = predict(model,s=best.lambda,as.matrix(test),type="response")


sig_SNP_LASSO = data.frame(which(coef(model)[, 10]!=0))
sig_SNP_LASSO = rownames(sig_SNP_LASSO)[2:nrow(sig_SNP_LASSO)]
tr_signif_LASSO = tr_signif[,sig_SNP_LASSO]

## REMOVE MORE
# PCA
library(logisticPCA)
library(ggplot2)
clogpca_cv = cv.clpca(tr_signif, ks = 2, ms = seq(100,600,100))
plot(logpca_cv)
# sig_SNP2 = sig_SNP[summary(model)$coefficients[,4]<0.05]
# 
# tr_signif2 = tr[,sig_SNP2]
# val=val[,sig_SNP2] # keep only significant features
# 
# df = data.frame(tr_signif2,tr_out) # avoid predict bugs of length
# model2 <- glm(tr_out ~ .,data=df,family="binomial")
# pred_tr = predict(model2,tr_signif2,type="response")
# 
# pred_val = predict(model2,newdata = val,type="response")




########  GPAS ?? #########
# library("RFreak")
# 
# 
# test_GPAS <- GPASInteractions(as.vector(tr_out), as.matrix(tr_signif), runs = 1, generations = 1000,occurences=10,ratio=0.1)


########  AUC #########
library("pROC")
auc(tr_out, pred_tr)
# auc(tr_out, pred_tr[,2]) # random forest
auc(tr_out_val, pred_val)
# auc(tr_out_val, pred_val[,2])


pred_te = data.frame(pred_te)
colnames(pred_te) <- "TARGET"


######### SAVE AS CSV FOR PYTHON SGD CLASSIFIER ###############
write.csv(tr_signif,file="CSV/tr_signif_10864.csv",row.names=FALSE)
write.csv(tr_signif_LASSO,file="tr_signif_LASSO.csv",row.names=FALSE)

write.csv(tr_out,file="tr_out_18.csv",row.names = FALSE)
write.csv(val,file="CSV/val_10864.csv",row.names = FALSE)
write.csv(tr_out_val,file="val_out_18.csv",row.names = FALSE)
write.csv(pred_te,file="my_pred_LASSO_9122.csv",row.names = FALSE)


write.csv(test,file="CSV/test_10864.csv",row.names = FALSE)
