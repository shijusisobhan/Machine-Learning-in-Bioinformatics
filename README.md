# Application of Regularization Technique in Bioinformatics : A Real World Example of Machine Learning 
Technological advances in next-generation sequencing (NGS) generate massive amounts of data that allowed researchers to study wide genetic variation and gene expressions. The explosion of these omics data challenges the existing methodologies for data analysis. With the increasing availability of different types of omics data, the application of machine learning (ML) methods has become more persistent. The machine-learning method provides a powerful tool for classifying and observing biological data. The aim of this field of bioinformatics is to develop computer algorithms capable of identifying predictable patterns in biological data samples. In this tutorial, I am introducing the applications of regularization techniques in ML to solve biological problems. Here I am presenting a test example and a real-world example with R code. Data and codes are upload in the github page. 

## Prerequisites
1. You are familiar with basic machine learning techniques (eg: Linear Regression)
2. You have basic knowledge in R programming 

## Work flow
1. Introduction to basic terminologies in ML and Regularization
2. Application of regularization with a synthetic data (with R code)
3. Introduction to biological background of the real-world problem
4. Application of regularization to solve the real-world problem (with R code)

## Introduction to basic terminologies in ML and Regularization
### Bias and variance
Bias- The inability of a ML technique to capture the true relation is called bias
Eg: We are trying to fit a curve data with linear regression model. It never captures the curve relationship. Straight line has very little flexibility. So, it has high bias. In contrast squiggly line have lower bias and it capture it easily.
Variance – Differences in fits between data points are called variance. 
For A good ML model should have low bias and low variance. 

### Regularization
Main goal of regularization is avoiding overfitting. Overfitting means the model is trying too hard to capture the data points that don’t really represent the true properties of your data. End result of over fitting is low accuracy. One of the ways of avoiding overfitting is using cross validation that helps in estimating the error over test set, and in deciding what parameters work best for your model.  
See the data below which shows the relationship between number of rooms and the cost of the house. Since the data look liner, we can use linear regression (least square) by minimizing the sum of mean square error (MSE):

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Overfit.jpg?raw=true" width="500"> 
*Fig 1*


$cost= m * (# Rooms) + c, given   MSE = ∑(actual cost-predicted cost)^2 is minimum ----(1)

If we have many measurements, least square accurately predict the relationship between cost and number of rooms (Green line). On the other hand, if we have only fewer measurements (only two red dots), we fit a new line with least squares (Black thick line). We can call red dots are training data and remaining blue dots are test data. Since newline overlap the two red data points, MSE for training data is zero, however MSE for testing data is large (High variance). That means the new line is overfit to the training data. So, the main idea behind the regularization is to find a new line that doesn’t fit the training data as well. It reduces the variance by introducing small amount of bias (Black dashed line).  Different regularization techniques are

1. Ridge regression - It minimizes ∑(actual cost-predicted cost)^2 + λ (m^2) to fit the line
2. Lasso regression - It minimizes ∑(actual cost-predicted cost)^2 + λ |m| to fit the line
3. Elastic net regression – It is a combination of ridge regression and lasso regression. It minimizes ∑(actual cost-predicted cost)^2 + λ (m^2) + λ |m| to fit the line

Where λ is the ridge/lasso/elastic net regression penalty

Now again go back to the equation-1, ie, $cost= m * (# Rooms) + c, where we need to estimate two parameters m and c. In this case, we need at least two data point to estimate those parameters. If we have only one data points, then we wouldn’t be able to estimate those parameters. Look at the Fig-2A, If there is only single point, you can fit any line passing through it, however, there is no way to tell if the red line is better than green line or black line or any other line goes through this data point. If we have two data points then it is clear that which the least square solution (Fig 2B).

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Singlepoint.jpg?raw=true" width="800"> 
*Fig 2*

Now let’s look at the case where we have two variables (#Rooms, #Bathrooms) and need to estimate 3 parameters. The model equation is as follows:


$cost= m1 * (# Rooms) +m2* (#Bathrooms)+ c


In this case, 2 data points is not sufficient to estimates the 3 parameters (m1,m2,c), instead, we required at least 3 data points. Likewise, if have n variable, then we have to estimate n+1 parameters and minimum n+1 data point is required to estimates those parameters. If we have an equation with 5000 variables, then we need at least 5001 data points to estimate those parameters. An equation with 5000 variables might sound crazy, but it is more common in real life. For example, we may use 5000 gene expression to predict some disease (eg. Cancer) and that means we have to collect gene expression data from 5001 human. However, collecting gene expression from 5001 person is time consuming and very expensive. Practically we can collect data from 500 or 1000 humans. What we do if we have a ML model with 5001 variables and only 1000 or less than 1000 data points. We use regularization techniques in ML!!!. It can find a solution with cross validation and λ that favors smaller parameter value. 


Another problem in ML is to identify the correct variables in data, which has true correlation with output. For example, table-1 shows the prices of the house. It has 4 variables, among them house number doesn’t have any relation with house price.  It is obvious, and by looking the data we can identify it. But what will happens if we don’t know about it? This problem also can be solved by using regularization method, which will identify the exact variables, which has true relationship with output.

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Data.jpg?raw=true" width="800"> 

### Multi response models
ML model to learn predictive relationships between multiple input and multiple output variables. For example, assume the table-1 have one more column, house insurance premium/year, which is also a function of the same input variables. Simply we can say that multiple output can be predicted from the same set of input as follows.


y1 = m11 * x1 + m12 * x2 + c1


y2= m21 * x1 + m22 * x2 + c2


<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/matrix.jpg?raw=true" width="200">

## Multi response elastic net regression Example-1 (synthetic data)

Problem overview 


1. We are generating a 5000 gene expression for 1000 data sample
2. Define two output variables, which are depending on input gene expressions, but only to a selected number of genes.
3. Apply elastic net regression to predict those output from the gene expression data and predict the exact genes which are really correlate with the output. 


```markdown
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
x
```
First 10 row and column of x (Total size 1000 x 5000)


<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/xdata.jpg?raw=true" width="600">


```markdown
## Define the output y1 and y2 (multi response)

# Index of the Genes that actually correlate with y1
real_p1=c(1,3,5,7,9,11,13,15,17,19,21,23,25,27,29) 
# Index of the Genes that actually correlate with y1
real_p2=c(21,24,25,29,30,35,40,45,60,61,68,90,99)
y1=apply(x[,real_p1], 1, sum)+rnorm(n)
y2=apply(x[,real_p2], 1, sum)+rnorm(n)
y=data.frame(y1,y2)
y
```
First 10 row of y (Total size 1000 x 2)


<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/ydata.jpg?raw=true" width="200">


```markdown

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
```

```markdown

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
```
