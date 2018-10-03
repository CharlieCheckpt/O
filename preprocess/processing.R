setwd("~/Documents/Challenge/AWS")
# load data

collapse_snip_pairs <- function(X){
  N = nrow(X)
  M = ncol(X)
  print(N)
  print(M)
  half_X = matrix(nrow=N, ncol=(M/2))
  print(dim(half_X))
  
  seq1 = seq(1,M-1,2)
  
  ind = 1
  for(i in seq1){
    half_X[,ind] = X[,i] + X[,i+1]
    ind = ind+1
  }
  
  return(half_X)
}


library("data.table")
Xtr = fread("./data/Xtrain_challenge_owkin.csv",data.table = FALSE) 
Xte  = fread("Xtest_challenge_owkin.csv",data.table = FALSE)
Ytr = fread("./data/challenge_output_data_training_file_disease_prediction_from_dna_data.csv",data.table = FALSE)

print("data loaded")

Xtr = Xtr[,-1]
Xte = Xte[,-1]
Ytr = Ytr[,-1]

print(dim(Xtr))
print(dim(Xte))
print(dim(Ytr))

print("collapsing ...")

Xtr = collapse_snip_pairs(Xtr)
half_Xte = collapse_snip_pairs(Xte)

print(dim(half_Xtrain))
print(dim(half_Xtest))

print("collapsed")

save(half_Xtrain,file="./data/Xtrain_challenge_owkin_half.Rda")
save(half_Xtest,file="./data/Xtest_challenge_owkin_half.Rda")
save(Ytr,file="./data/Ytrain_challenge_owkin_half.Rda")

print("new matrices saved")


# 
setwd("~/Documents/Challenge/AWS")
load("./data_Rda/Xtrain_challenge_owkin_half.Rda")
load("./data_Rda/Ytrain_challenge_owkin_half.Rda")
load("./data_Rda/Xtest_challenge_owkin_half.Rda")
