
# This is a simulation study to demonstrate:
# Elastic net regression on NGS data

# Clear the global environment
rm(list=ls())

# To Reproduce the same output of simulation study
set.seed(200) 

# Load the required package to perform elastic net regression
library('glmnet')

n=1000 # Number of observations (sample)
p=5000 # Number of genes

# Difine Input (x-->gene expression data)
x=matrix(rnorm(n*p), nrow = n, ncol = p)
nn<-1:5000
colnames(x)<-paste('gene', nn, sep = '.')

## Define the output y1 and y2 (multi response)

# Index of the Genes that actually correlate with y1
real_p1=c(1,3,5,7,9,11,13,15,17,19,21,23,25,27,29) 
# Index of the Genes that actually correlate with y1
real_p2=c(21,24,25,29,30,35,40,45,60,61,68,90,99)
y1=apply(x[,real_p1], 1, sum)+rnorm(n)
y2=apply(x[,real_p2], 1, sum)+rnorm(n)
y=data.frame(y1,y2)

### ******split data into train and test *************************************
train_rows<-sample(1:n, 0.66*n)

x_train<-x[train_rows,]
x_test<-x[-train_rows,]

y_train<-y[train_rows,]
y_test<-y[-train_rows,]

y_train_1<-data.matrix(y_train)
y_test_1<-data.matrix(y_test)

## *************Train the data with train data ***********************************

Elastic_net_regg<-cv.glmnet(x_train, y_train_1, keep=T, alpha=0.5, family='mgaussian')

## Predict the output using test data

Pred_net<-(predict(Elastic_net_regg, s=Elastic_net_regg$lambda.1se, newx = x_test))

# calculate the means square error between predicted output and test output
Pred_net<-data.matrix(Pred_net)
y_test_1<-as.vector(y_test_1)
mse_Net<-mean((Pred_net-y_test_1)^2)
mse_Net 

## ****plot test output vs predicted output ************

plot(y_test_1, Pred_net)

## ******Extract the real genes that correlate with output **********************

P.coef<-(predict(Elastic_net_regg,type="coef")) # Coeeficient associated with each genes

#Real genes that corelate with y1
P_y1<-as.data.frame(P.coef$y1[row.names(P.coef$y1), ]) # Extract coefficient for y1
colnames(P_y1)<-'coefficients'
P_y1$genes<-row.names(P_y1) # Extract gene names
P_y1<-P_y1[,c(2,1)]
P_y1_sort<-P_y1[order(-P_y1$coefficients),] # sort genes with decreassing values of coefficient

#  Here we consider coefficient associated with genes have a value >1e-1 (0.01) as real genes
sig_genes_y1<- P_y1_sort[which(P_y1_sort$coefficients > 1e-1),] 
sig_genes_y1

#Real genes that corelate with y2
P_y2<-as.data.frame(P.coef$y2[row.names(P.coef$y2), ])
colnames(P_y2)<-'coefficients'
P_y2$genes<-row.names(P_y2)
P_y2<-P_y2[,c(2,1)]
P_y2_sort<-P_y2[order(-P_y2$coefficients),]
sig_genes_y2<- P_y2_sort[which(P_y2_sort$coefficients > 1e-1),]
sig_genes_y2


