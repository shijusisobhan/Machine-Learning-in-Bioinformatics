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

<img src="https://github.com/shijusisobhan/Machine-Learning-in-Bioinformatics/blob/main/Figures/Overfit.jpg?raw=true" width="400">

$cost= m * (# Rooms) + c, given   MSE = ∑(actual cost-predicted cost)^2 is minimum ----(1)

If we have many measurements, least square accurately predict the relationship between cost and number of rooms (Green line). On the other hand, if we have only fewer measurements (only two red dots), we fit a new line with least squares (Black thick line). We can call red dots are training data and remaining blue dots are test data. Since newline overlap the two red data points, MSE for training data is zero, however MSE for testing data is large (High variance). That means the new line is overfit to the training data. So, the main idea behind the regularization is to find a new line that doesn’t fit the training data as well. It reduces the variance by introducing small amount of bias (Black dashed line).  Different regularization techniques are

1. Ridge regression - It minimizes ∑(actual cost-predicted cost)^2 + λ (m^2) to fit the line
2. Lasso regression - It minimizes ∑(actual cost-predicted cost)^2 + λ |m| to fit the line
3. Elastic net regression – It is a combination of ridge regression and lasso regression. It minimizes ∑(actual cost-predicted cost)^2 + λ (m^2) + λ |m| to fit the line

