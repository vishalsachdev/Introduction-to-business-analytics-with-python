#!/usr/bin/env python
# coding: utf-8

# # Multiple Linear Regression

# ## Chapter Introduction
# In the previous chapter, we discussed similarity and clustering as a data pre-processing step. It is an important organizational step that helps analyze the data as a whole before moving into regression analysis and identifying the direct impact of certain variables on others. In this chapter, we’ll go over the basics of that regression analysis, focusing on one of the most popular models for making predictions when we have a numerical outcome variable, Multiple Linear Regression (MLR). 
# 
# In this chapter, we’ll go over: 
# 1. What is MLR?
# 2. Assumptions behind MLR
# 3. Steps to MLR
# 4. MLR + Predictions

# ### What is MLR
# 
# ![Title](img/MLR/image31.png)

# Recall from Chapter 2 that data mining methods fall under two main categories: supervised learning and unsupervised learning. These learning methods are classified based on the presence and type of outcome variable; in supervised learning, we predict a previously determined outcome variable, and in unsupervised learning, we identify patterns within the data. Under unsupervised learning, the main method we’ll cover in this class is clustering, and under supervised learning, the two branches we’ll cover are regression and classification.
# 
# The goal of **Regression** is to predict numerical outcomes and can be used to predict values in various situations. For example, predicting the amount spent on a fraudulent transaction would be a regression problem.
# 
# Regression can be used for explanatory or predictive modeling.
# 
# **Explanatory Modeling** - Creating a model that fits well with existing data (the data on which it was trained)
# Uses the entire data set to maximize information and focus on the coefficients
# Represents the relationship between existing dependent and independent variables, not predicting new values for dependent variables
# 
# **Predictive Modeling** – Creating a model that has high predictive accuracy on new data
# Partitions to assess predictive performance and avoids overfitting with the focus being on predictions
# Accurately predicts new values for dependent variables, not descriptions of associations between existing dependent and independent variables 
# 
# The goal of **Classification** is to predict categorical outcomes, such as whether or not an outcome will be true or the probability that an outcome will occur. Within classification, there are two types of outcomes: class membership and propensity. Class membership refers to the categorical variable itself, while propensity refers to the probability of the class membership. For example, predicting whether or not a transaction is fraudulent would be a classification problem.
# 
# Here are some more examples:
# 
# How much will a house sell for? - Regression (numerical outcome)
# 
# Will a student pass a test or not? - Classification (class membership)
# 
# What is the probability of a certain student passing a class? - Classification (propensity)
# 
# One method of predicting numerical outcomes under regression is Linear Regression, which examines the relationship between a dependent variable (also referred to as the outcome variable, response, or target) and independent variable(s) (also referred to as the predictor(s), input variable(s), regressor(s), or covariate(s)) assuming a linear relationship. Simple Linear Regression (SLR) examines a single independent variable, while Multiple Linear Regression (MLR) examines multiple independent variables.
# 
# The equation below represents a linear relationship between multiple predictors and the outcome. We assume that this is the true relationship between the predictor variables and outcome variable. We will go over some examples and explanations to the different parts of this equation in the rest of this chapter.
# 
# y = β0 + β1x1 + β2x2 +⋯+ βpxp + ϵ
# 
# y: outcome variable
# x1, ..., xp: predictor variables wherewith p is the= total number of predictors
# β0: intercept
# β1, ..., βp: coefficients with p = total number of predictors
# ϵ: error (factors affecting y other than predictors)

# The above equation is a generic model for the relationship between multiple predictor variables and the outcome variable. However, in explanatory modeling, the focus is on the existing coefficients (β), and in predictive modeling, the focus is on predicting the new outcome variable (Y).
# 
# The remainder of this chapter will focus on predictive modeling, the more common ____ of the two in data mining. Recall that in a predictive MLR model, we are predicting new values for the coefficients, or betas, to create a model that has high predictive accuracy on new data. So, we estimate the betas () using ordinary least squares (OLS). This method minimizes the sum of squared deviations between the actual and predicted outcome variable values (Y and Y) based on the given model.

# ## Assumptions about MLR

# When dealing with multiple linear regression, we must always assume that there is a linear relationship between multiple predictors and the outcome. If a linear relationship is not present, a non-linear regression model must be utilized.
# 
# We must also assume independent variables are not highly correlated to one another; this creates the issue of multicollinearity. 
# 
# When converting categorical variables to dummy variables, we must drop one dummy variable to avoid multicollinearity. For example, continuing on from the Toyota dealership example, if we were to dummy code the variable Fuel, we would get Fuel_Petrol, Fuel_Diesel, and Fuel_CNG. If Fuel_Patrol and Fuel_Diesel are both 0, then the car in question must run on CNG. Including the variable Fuel_CNG in our model means that any one of the Fuel dummy variables can be determined by the remaining two dummy variables.This is an example of perfect multicollinearity. Therefore, as shownyou see in the image above, Fuel_CNG must be dropped from the model. It doesn’t matter which dummy variable is dropped, as long as one of them is.
# 
# By fitting the model to our data, we will estimate the coefficients to the predictor variables and use those to predict outcome values. We can then use those predicted outcome values to calculate residuals (also called errors). Estimates will remain unbiased if the mean of the error term is 0, and if the variance of the error term is constant, or homoskedastic.

# ![Title](img/MLR/image50.png)

# ## MLR + Prediction

# In data mining, predictive accuracy methods indicate high accuracy when applied to new data sets, not existing ones. There are 5 common predictive accuracy methods we will discuss in this chapter. These include Mean Error (ME), Root-Mean-Squared Error (RMSE), Mean Absolute Error (MAE), Mean Percentage Error (MPE), and Mean Absolute Percentage Error (MAPE). 

# ![Title](img/MLR/image35.png)

# **Mean Absolute Error (MAE)**: This measure gives the absolute magnitude of the distance between your prediction and the outcome variable. For example, if your MAE is 12, your predictions are on average 12 units off from the actual value. 
# 
# **Mean Error (ME)**: This measure gives an indication of whether predictions overpredict or underpredict the outcome variable on average. It is similar to MAE, but does not take the absolute value of the error. A small mean error does NOT indicate good performance, but tells us whether the model consistently over- or underestimates outcomes. 
# 
# **Root-Mean-Squared Error (RMSE)**: This measure tells you how concentrated the data is around the line of best fit determined by the regression method. It is similar to the standard error of estimate in linear regression, but computed on the validation data. 
# 
# **Mean Absolute Percentage Error (MAPE)**: This measure gives the percentage score of how predictions deviate from the actual values on average. 
# 
# **Mean Percentage Error (MPE)**: This measure is similar to MAPE, but does not take the absolute value of the error.

# ## MLR Steps (Predictive Modeling)

# In Predictive MLR, the data are split into two sets or partitions: training and validation (also called holdout set). The training set is used to get predicted beta values and estimate the model, and the validation set is used to assess performance on new data. We first split the data into training and validation sets, using given percentages (eg: 60% and 40%). Then, we use the training set to fit a multiple linear regression model between the outcome variable and the other variables. Recall from Chapter 1 and earlier in this chapter that we create dummy variables for each predictor variable and drop 1 to avoid multicollinearity. We then apply the predicted beta values to the test set to get predicted y values, which are then compared to the actual y values to get error values and predictive accuracies. Lastly, we can use the predictive accuracy methods outlined above to choose the best model. 
# 
# Split data into training and validation sets
# Apply MLR to the training partition to get predicted 𝛽0 …, 𝛽p values, or 
# The MLR algorithm automatically does this through minimizing the sum of squared residuals for the training data
# Apply predicted 𝛽0 ..., 𝛽p values to the test set to get predicted y values, or y
# Compare predicted y values to actual y values to get error values and predictive accuracy
# Compare models of different predictive accuracies to determine the best model

# ## MLR Example

# Let’s walk through an example of predicting numerical outcomes.
# 
# You work at a Toyota dealership which buys back used cars from customers purchasing new cars. To make a profit, the dealership needs to predict the price that the used car will sell for.

# | Variable   | Description                           |
# |------------|---------------------------------------|
# | Price      | Offer price in Euros                  |
# | Age        | Age in months as of August 2004       |
# | Kilometers | Accumulated Kilometers on odometer    |
# | Fuel type  | Fuel type (Petrol, Diesel, CNG)       |
# | HP         | Horsepower                            |
# | Metallic   | Metallic color (Yes = 1, No= 0)       |
# | Automatic  | Automatic (Yes = 1, No = 0)           |
# | CC         | Cylinder volume in cubic centimeters  |
# | Doors      | Number of doors                       |
# | QuartTax   | Quarterly road tax in Euros           |
# | Weight     | Weight in Kilograms                   |

# Example: 
# 
# In this example, a relationship between several variables and price is already given. Later in this chapter, we’ll explore how to get these coefficients through MLR.
# 
# Price = 3 - 0.5 Age + 2 Automatic_binary + 3 PowerWindow_binary + 0.4 WarrantyLeft
# 
# We can interpret these variables quantitatively and qualitatively. A quantitative interpretation of this equation would state a 1 unit increase in the age of a car reduces price by 0.5 units of price whereas a qualitative interpretation of this equation would simply state an increase in age will reduce the price of the car (without specifying numerical values.)
# 
# We will use predictors (x1, ..., xp) to predict (instead of explain) outcome (y).
# 
# For example:
# -Use house characteristics to predict selling price
# -Use advertising expenditures to predict sales
# -Use product demand to predict valuation

# ![Title](img/MLR/image16.png)

# In[1]:


import pandas as pd
df = pd.read_csv("./data/ToyotaCorolla.csv")


# In[2]:


#Example
from sklearn.model_selection import train_test_split

X = df.drop('Price',axis = 1)
Y = df.Price

#Encoding variables
X = pd.get_dummies(X,drop_first = False)

#Partition
train_x,test_x, train_y, test_y = train_test_split(X,Y, test_size = 0.4, random_state = 2012)


# In[3]:


from sklearn.linear_model import LinearRegression

car_lm = LinearRegression()

car_lm.fit(train_x, train_y)

print('intercept ', car_lm.intercept_)
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': car_lm.coef_}))


# In[4]:


from dmba import regressionSummary

regressionSummary(train_y, car_lm.predict(train_x))
print()
regressionSummary(test_y, car_lm.predict(test_x))


# In[5]:


car_lm_pred = car_lm.predict(test_x) # Which data set are we calculating predictions
result = pd.DataFrame({'Predicted': car_lm_pred, 'Actual': test_y,'Error': test_y - car_lm_pred})
result.head(20)


# In[6]:


#Plotting Actual Vs Predicted
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

plt.scatter(result.Predicted,result.Actual)
plt.xlabel("Predicted Response")
plt.ylabel("Actual Values")
plt.plot( [5000,30000],[5000,30000],color="red" )

