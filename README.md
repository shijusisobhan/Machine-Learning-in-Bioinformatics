# Application of Regularization Technique in Bioinformatics: A Real-World Example of Machine Learning 
Technological advances in next-generation sequencing (NGS) generate massive amounts of data that allowed researchers to study wide genetic variation and gene expressions. The explosion of these omics data challenges the existing methodologies for data analysis. With the increasing availability of different types of omics data, the application of machine learning (ML) methods has become more persistent. The machine-learning method provides a powerful tool for classifying and observing biological data. The aim of this field of bioinformatics is to develop computer algorithms capable of identifying predictable patterns in biological data samples. In this tutorial, I am introducing the applications of regularization techniques in ML to solve biological problems. Here I am presenting a test example and a real-world example with R code. Data and codes are uploaded to the GitHub page. 

## Prerequisites
1. You are familiar with basic machine learning techniques (eg: Linear Regression)
2. You have basic knowledge of R programming 

## Workflow
1. Introduction to basic terminologies in ML and Regularization
2. Application of regularization with synthetic data (with R code)
3. Introduction to biological background of the real-world problem
4. Application of regularization to solve the real-world problem (with R code)

## Introduction to basic terminologies in ML and Regularization
### Bias and variance
Bias- The inability of an ML technique to capture the true relation is called bias
Eg: We are trying to fit curve data with a linear regression model. It never captures the curve relationship. A straight line has very little flexibility. So, it has a high bias. In contrast, squiggly lines have lower bias and capture it easily.
Variance – Differences in fits between data points are called variance. 
A good ML model should have low bias and low variance. 

### Regularization
The main goal of regularization is to avoid overfitting. Overfitting means the model is trying too hard to capture the data points that don’t represent the true properties of your data. The result of overfitting is low accuracy. One of the ways of avoiding overfitting is using cross-validation which helps in estimating the error over the test set, and in deciding what parameters work best for your model.  
See the data below which shows the relationship between the number of rooms and the cost of the house. Since the data look liner, we can use linear regression (least square) by minimizing the sum of mean square error (MSE): 

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Overfit.jpg?raw=true" width="500"> 
*Fig 1*


$cost= m * (# Rooms) + c, given   MSE = ∑(actual cost-predicted cost)^2 is minimum ----(1)

If we have many measurements, the least square accurately predicts the relationship between cost and the number of rooms (Green line). On the other hand, if we have only fewer measurements (only two red dots), we fit a new line with the least squares (Black thick line). We can call red dots training data and the remaining blue dots are test data. Since newlines overlap the two red data points, MSE for training data is zero, however, MSE for testing data is large (High variance). That means the new line is overfitted to the training data. So, the main idea behind the regularization is to find a new line that doesn’t fit the training data as well. It reduces the variance by introducing a small amount of bias (Black dashed line).  Different regularization techniques are

1. Ridge regression - It minimizes ∑(actual cost-predicted cost)^2 + λ (m^2) to fit the line
2. Lasso regression - It minimizes ∑(actual cost-predicted cost)^2 + λ |m| to fit the line
3. Elastic net regression – It is a combination of ridge regression and lasso regression. It minimizes ∑(actual cost-predicted cost)^2 + λ (m^2) + λ |m| to fit the line

Where λ is the ridge/lasso/elastic net regression penalty

Now again go back to equation-1, ie, $cost= m * (# Rooms) + c, where we need to estimate two parameters m and c. In this case, we need at least two data points to estimate those parameters. If we have only one data point, then we wouldn’t be able to estimate those parameters. Look at Fig-2A, If there is only a single point, you can fit any line passing through it, however, there is no way to tell if the red line is better than a green line or black line or if any other line goes through this data point. If we have two data points then it is clear which the least square solution (Fig 2B).

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Singlepoint.jpg?raw=true" width="800"> 
*Fig 2*

Now let’s look at the case where we have two variables (#Rooms, #Bathrooms) and need to estimate 3 parameters. The model equation is as follows:


$cost= m1 * (# Rooms) +m2* (#Bathrooms)+ c


In this case, 2 data points are not sufficient to estimate the 3 parameters (m1,m2,c), instead, we required at least 3 data points. Likewise, if have an n variable, then we have to estimate n+1 parameters, and a minimum n+1 data point is required to estimate those parameters. If we have an equation with 5000 variables, then we need at least 5001 data points to estimate those parameters. An equation with 5000 variables might sound crazy, but it is more common in real life. For example, we may use 5000 gene expressions to predict some disease (eg. Cancer) and that means we have to collect gene expression data from 5001 humans. However, collecting gene expression from 5001 people is time-consuming and very expensive. Practically we can collect data from 500 or 1000 humans. What do we do if we have an ML model with 5001 variables and only 1000 or less than 1000 data points? We use regularization techniques in ML!!!. It can find a solution with cross-validation and λ that favors smaller parameter values. 


Another problem in ML is to identify the correct variables in data, which has a true correlation with output. For example, table-1 shows the prices of the house. It has 4 variables, among them house number doesn’t have any relation with house price.  It is obvious, and by looking at the data we can identify it. But what will happen if we don’t know about it? This problem also can be solved by using the regularization method, which will identify the exact variables, which has a true relationship with output.

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Data.jpg?raw=true" width="800"> 

### Multi response models
ML model to learn predictive relationships between multiple input and multiple output variables. For example, assume the table-1 has one more column, house insurance premium/year, which is also a function of the same input variables. Simply we can say that multiple outputs can be predicted from the same set of inputs as follows.


y1 = m11 * x1 + m12 * x2 + c1


y2= m21 * x1 + m22 * x2 + c2


<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/matrix.jpg?raw=true" width="200">

## Multi response elastic net regression Example-1 (synthetic data)

Problem overview 


1. We are generating a 5000 gene expression for 1000 data sample
2. Define two output variables, which are depending on input gene expressions, but only on a selected number of genes.
3. Apply elastic net regression to predict those outputs from the gene expression data and predict the exact genes which are really correlated with the output. 


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
<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/synthetic_pred_vs_test.jpg?raw=true" width="400">

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



## Real-world example: Predict physiological time based on gene expression in human blood

### A brief introduction to the biological background and relevance of the problem

Circadian rhythm (Body clock) – Circadian rhythms are internally driven cycles of hormone\gene\ protein that rise and fall during the 24-hour day. It helps you fall asleep at night and wake you up in the morning. It simply maintains the timing of the internal body, so it is known as the body clock. Circadian clocks play a key role in regulating a vast array of biological processes, with significant implications for human health. Humans are heterogenous in internal body time. Accurate assessment of physiological time can significantly improve the diagnosis of circadian disorders and optimize the delivery time of therapeutic treatments.
The current method to assess internal time is dim-light melatonin onset (DLMO): Determine the time point when endogenous melatonin reaches a predefined threshold concentration in blood plasma. The major limitation of this technique is the need for serial sampling over extended periods of several hours, which is both costly and burdensome to the patient. So main aim of this problem is to predict physiological time based on gene expression in human blood. [Braun et al. (2018)](https://pubmed.ncbi.nlm.nih.gov/30201705/) Introduce machine learning techniques to predict the internal timing from human gene expression data. The basic ideas behind the algorithms are subject-wise normalization of the gene expression data and elastic net regression. Those who have interested in this research work please see the link. Here I provide the different steps involved in the technique.

1. log2 normalization of gene expression data
2. normalization of data within the subject

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Normalization.jpg?raw=true" width="200">

3. Collect the time after melatonin reaches 25% (DLMO25). Convert the time into angle of the hour
hand on a 24-h clock. θi =2*pi*ti/24, where ti is the time point when endogenous melatonin reaches 25% concentration in blood plasma.
4. Perform a bivariate regression of the cartesian coordinates (The melatonin level with respect to the time roughly follows a cosine wave with a period of 24hr)

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Regression.jpg?raw=true" width="400">

5. In this problem the number of predictor variables, ie genes (7616) are larger than number of observations (355). Also, the majority of the genes will not have strong relationship with internal time. So, to reduce the overfitting and obtain a simple model, elastic net regularization for feature selection. It solves following equation to get the best fit.

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/elastic.jpg?raw=true" width="400">


Here we demonstrate the Implementation of the Machine learning algorithm on R. The data used here is originally presented by [Moller et. al (2013)](https://pubmed.ncbi.nlm.nih.gov/23440187/). I have downloaded the data and normalized it based on the techniques described in [Braun et.al (2018)](https://pubmed.ncbi.nlm.nih.gov/30201705/). I already uploaded the normalized gene expression data on the GitHub page.

```markdown
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

```
<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/pred_vs_test_reala_data.jpg?raw=true" width="400">

Elastic nete regression accurately prdicts the internal body time from gene expression data. Note that the data points on the two corners are not an outlier or bad results, instead, they are also show high accuracy, because the true time is modulo 24.
```markdown

## ******Extract the real genes that correlate with DLMO25 time **********************

P.coef<-(predict(Ridge_regg,type="coef"))
P_x<-as.data.frame(P.coef$hrs_after_DLMO25[row.names(P.coef$hrs_after_DLMO25), ])
colnames(P_x)<-'coefficients'
P_x$genes<-row.names(P_x)
P_x<-P_x[,c(2,1)]
Px_sort<-P_x[order(-P_x$coefficients),]
sig_genes<- Px_sort[which(Px_sort$coefficients > 0),]
sig_genes
```

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Sig_genes.jpg?raw=true" width="300">

Significant genes in one iteration are shown here. User can try it with different training sets and different threshold. In the litterature, [Braun et.al (2018)](https://pubmed.ncbi.nlm.nih.gov/30201705/) listed a set of 41 genes which is obtained via 12 repeated runs using different training samples.
