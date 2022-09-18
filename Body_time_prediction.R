
# Clear the global environment
rm(list=ls())
set.seed(200) 

# set the path where code and data stored. Please change this line according to your local machine path
setwd('D:/Github_files/Machine-Learning-in-Bioinformatics')

# Load the required package to perform elastic net regression
library('glmnet')

## Load the data
gene.expr<-read.csv('Moller_Normalized_gene_expression.csv')
DLMO25<-read.csv('Moller_DLMO25_data.csv')

## *********  Define x (Gene expression data)**********************
x<-gene.expr[-1]

##******** Define y (Hrs after DLMO25)********************

#Conver time into angles --->   a=2*pi*time/24

DLMO25_angle<-(DLMO25[-1]%%24)*2*pi/24

# Convert angle into cartesian coordinates   
y1<-sin(DLMO25_angle) # sin(a)
y2<-cos(DLMO25_angle) # cos(a)
y=data.frame(y1,y2) # y= [sin(a) cos(a)]

### ******split data into train and test *************************************
n=nrow(y)
train_rows<-sample(1:n, 0.66*n)

x_train<-x[train_rows,]
x_test<-x[-train_rows,]

x_train<-data.matrix(x_train) # Just for data handling convert it into matrix
x_test<-data.matrix(x_test)

y_train<-y[train_rows,]
y_test<-y[-train_rows,]

y_train<-data.matrix(y_train)
y_test<-data.matrix(y_test)

## *************Train the data with train data ***********************************

Ridge_regg<-cv.glmnet(x_train, y_train, keep=T, alpha=0.5, family='mgaussian')

## **********Predict the output using test data *********************************

Pred_net<-(predict(Ridge_regg, s=Ridge_regg$lambda.1se, newx = x_test))
Pred_angle<-Pred_net[,,1] # Extract the predicted angle
Pred_Time<-(atan2(Pred_angle[,1],Pred_angle[,2])%%(2*pi))*(24/(2*pi)) # convert back angle to time
y_test_Time<-(atan2(y_test[,1],y_test[,2])%%(2*pi))*(24/(2*pi))

## Find the mean square error *************************

MSE_time<-mean((Pred_Time-y_test_Time)^2)
MSE_time

## ****plot original time vs predicted time ************

plot(y_test_Time, Pred_Time)

## ******Extract the real genes that correlate with DLMO25 time **********************

P.coef<-(predict(Ridge_regg,type="coef"))
P_x<-as.data.frame(P.coef$hrs_after_DLMO25[row.names(P.coef$hrs_after_DLMO25), ])
colnames(P_x)<-'coefficients'
P_x$genes<-row.names(P_x)
P_x<-P_x[,c(2,1)]
Px_sort<-P_x[order(-P_x$coefficients),]
sig_genes<- Px_sort[which(Px_sort$coefficients > 1e-1),]
sig_genes









