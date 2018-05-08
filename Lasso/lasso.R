
#  general parameters
test = FALSE # set to true if you want fast cv to see if script works
load_all = FALSE # set to true if you want to load data with all ft (36000)
load_model = FALSE # set to true if you want to reload previous model

# load data
print("loading data ...")
if(load_all){
  print("->loading all")
}else{
  print("-> loading half")
  tmp_x = load("./data/Xtrain_challenge_owkin_half.Rda")
  tmp_y = load("./data/Ytrain_challenge_owkin_half.Rda")
  df_x = get(tmp_x)
  df_y = get(tmp_y)
  tmp_x = load("./data/Xtest_challenge_owkin_half.Rda")
  df_x_te = get(tmp_x)
  rm(tmp_x)
  rm(tmp_y)
}

if(test){
  print("subsetting ...")
  df_x = df_x[1:100,1:5]
  df_y = df_y[1:100]
  df_x_te = df_x_te[1:50,1:5]
}

# put data to right "format"
Xtr = as.matrix(df_x)
Xte = as.matrix(df_x_te)
Ytr = factor(df_y)

rm(df_x)
rm(df_y)
rm(df_x_te)

print("train data loaded")
print(paste0("Xtr : ", dim(Xtr)))
print(paste0("Ytr : ", dim(Ytr)))

alpha_grid = c(0.5,0.8,1)

history = matrix(0, nrow = length(alpha_grid), ncol = 3)
rownames(history) = alpha_grid
colnames(history) = c("lambda","nnz_coeffs","cv_auc")

print("alpha space for grid search : ")
print(alpha_grid)

if(load_model==FALSE){
  # LASSO MODEL
  library(doParallel)
  library(glmnet)
  for(i in 1:length(alpha_grid)){
    a = alpha_grid[i]
    cat("Starting cv with alpha :",a," \n")
    registerDoParallel(10)
    cvfit <- cv.glmnet(Xtr, y= Ytr, alpha = a,family="binomial", type.measure = "auc", parallel=TRUE)
    cat("CV with alpha :",a," ended \n")

    best.lambda <- cvfit$lambda.min

    auc_cvfit = max(cvfit$cvm)
    coef.min = coef(cvfit, s = "lambda.min")
    active.min = which(coef.min != 0)
    index.min = coef.min[active.min]

    cat("number of non-zero coeffs : ", length(active.min), " \n")
    cat("lambda chosen : ", best.lambda, "\n")
    cat("cv auc : ",auc_cvfit,"\n")

    history[i,1] = best.lambda
    history[i,2] = length(active.min)
    history[i,3] = auc_cvfit
    write.table(history, file = "lasso_history_cv_on_alpha_lambda.csv")


    pred_te = predict(cvfit, newx = Xte, s = 'lambda.min', type="response")

    file_pred_name = paste0("pred_lasso_alpha=",a,"_lambda=",cvfit$lambda.min,"_cvfit.csv")
    write.table(pred_te,file=file_pred_name,row.names=FALSE)
    cat("preds saved as : ",file_pred_name,"\n")

  }

}else{
  print("loading model")
  load("cvfit_lasso_18000.rda")
  load('best.lambda_18000.rda')
  print("model cvfit loaded")
  cat("best lambda loaded : ", best.lambda,"\n")
}



#
# # TEST SET
# load("./data/Xtest_challenge_owkin_half.Rda")
#
# print("test data loaded")
# pred_te = predict(cvfit, newx = Xte, s = 'lambda.min', type="response")
#
# write.table(pred_te,file="pred_lasso_18000_R_cvfit.csv",row.names=FALSE)
# print("preds saved")
