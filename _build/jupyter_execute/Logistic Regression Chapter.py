#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# 
# 
# ## What is Logistic Regression? 
# 
# <span style="color:orange">**Logistic regression** </span>is a form of supervised learning that uses past data to predict the value of the outcome. Although it has “regression” in the name, logistic regression is actually used for classification.

# | ![Classification vs Regression](attachment:classification_vs_regression.png)|
# |:--:|
# |<b>Fig. # - Classification predicts discrete variables, like True/False, Yes/No. Regression predicts continuous variables, like Age, Salary, and Income.</b>|

# As discussed in Chapter X, the main difference between classification and regression deals with the dependent variable. When it comes to classification, you’re working with categorical variables. This means, you’re either predicting whether or not an observation belongs to a certain class, or the probability of your observation belonging to a specific class. For regression, on the other hand, you’re predicting a continuous, numerical amount. 
# 
# Since logistic regression models are used for classification, their goal is to determine the binary dependent variable. So, situations depicting logistic regression will only have **two potential outcomes** that are clearly distinguished. 
# 
# Some examples of logistic regression include: 
# 
# *   Identifying whether a tumor is malignant _or_ benign 
# *   Detecting whether an email is spam _or_ legitimate 
# *   Predicting whether a student will pass _or_ fail a class 
# 
# Now, let’s look at an example together. Imagine you’re working at a bank, and your job is to determine whether or not a customer will accept a loan. Whether or not the customer accepts the loan will be the outcome variable. Our predictors will be factors such as demographics, income, and the customer’s relationship with the bank. Let’s take this step by step. 

# | ![Logistic_reg_cutoff](attachment:logistic_reg_cutoff.png)|
# |:--:|
# |<b>Fig. # - In this logit graph, the orange line represents the cutoff point. This line will determine which class an individual belongs to. </b>|

# In the graph above, the orange line represents our cutoff point. So, if our input variable, income, falls below that orange cutoff line, it will equate to a value of ‘0’. This means that the customer will not accept the loan offer. On the flip side, if income is above that orange cutoff line, it will equate to a value of ‘1’. In that case, we would assume the customer would accept the loan offer.

# ### How is Logistic Regression Helpful? 
# 
# As you can see from the example above, there are a lot of benefits to using logistic regression models: 
# 
# 
# 
# *   They’re helpful when dealing with several predictor variables  
# *   They provide a clear and definitive answer (Yes/No, Pass/Fail, True/False, etc.)
# *   They can readily updated with new data
# 
# The main drawback with logistic regression, however, is its limited curve. Since it can only produce values between 0 and 1, these models are unable to showcase complex relationships. 

# ## The Logistic Function
# 
# $$
# log(odds) = \beta_{0} + \beta_{1}x_{1} + \beta_{2}x_{2} + ... + \beta_{p}x_{p} 
# $$
# 
# If you’re looking at this equation and thinking to yourself, _Hey, this kind of looks familiar_, then you’d be thinking correctly. If you recall from our past chapter about Multiple Linear Regression, the right half of this equation is the same as the equation for MLR. The only differences are that (1) instead of it being equal to Y, in this case, the function is equal to log(odds), and (2) the equation does not account for error, or E. log(odds) is also called a <span style="color:orange">**logit function**. </span> 
# 
# 
# ### Odds Ratio
# 
# Consider this scenario: you’re watching a game, where a highly skilled (Team A) is playing an amateur one (Team B), and Team A wins 4 out of 5 matches. So, Team A has won 4 games and lost 5. In this case, the odds of Team A winning are 4:1, while the probability of them winning is ⅘. Alternatively, Team B’s odds of winning are 1:4, while the probability of them winning is ⅕.
# 
# As you can see, **odds and probabilities are not the same**. An <span style="color:orange">**odds ratio** </span> can be defined as the likelihood of something occurring divided by the likelihood that it won’t. <span style="color:orange">**Probability** </span>, on the other hand, is defined as the number of favorable outcomes (in this case, Team A winning, if you’re rooting for Team A) divided by the total number of outcomes. Both are ways of expressing the likelihood of something happening; they’re just calculated differently. 
# 
# What’s nice about these two functions is that you can find one by calculating the other. Once you know the frequencies of Win/Loss, True/False, Yes/No, etc., you can calculate either odds or probabilities. 
# 
# $$
# Odds(Y=1) = p/(1-p)
# $$
# $$
# p = odds/(1+odds)
# $$
# 
# Furthermore, if the odds are against Team A winning, then they will be between 0 and 1 (ex. 1/32); if the odds Team A winning are good, then the odds will be between 1 and infinity (ex. 32/3).
# 
# Comparing these odds can be a bit difficult, but if we were to use a log function, it would be much clearer. The following example is inspired from the YouTube video, “StatQuest: Odds and Log(Odds), Clearly Explained!!!”, as linked [here](https://www.youtube.com/watch?v=ARfXDSkQf1Y). 
# 
# Let’s say the odds are 1 to 5. If the odds are against 1 to 5, then that means the odds are 0.2. On the flip side, that means the odds are in favor of 5 to 1, then the odds are 5. Obviously, the magnitude of the left side (in orange) is drastically different from the odds on the right side (blue). This makes the problem asymmetrical and hard to understand.

# | ![Asymmetrical Odds](attachment:prob_vs_odds_asymmetric.png)|
# |:--:|
# |<b>Fig. # - When problems are asymmetrical, they can become difficult to understand.</b>|

# So, as we learned in the StatQuest video, this is where logs can come in handy. If the odds against were 1 to 5, the log(odds) are -1.39, and if the odds are in favor 5 to 1, the log(odds) are 1.39. After using the log function to analyze the odds, we see the magnitude of the distance from the center remains the same. 
# 
# | ![Symmetrical Odds](attachment:prob_vs_odds_symmetric.png)|
# |:--:|
# |<b>Fig. # - By using a log function, it makes it easier to analyze the odds of a situation.</b>|

# Let’s go back to the logit function we just learned about for a second. 
# 
# In our last chapter about MLR, the lowercase “x” variables on the right side of the equation are called “predictors”. When it comes to linear functions, their outcome variables are continuous and can therefore have an infinite number of possible values. 
# 
# However, logistic functions are binary. So, the addition of the log in the equation ensures that predictor values (which measure probability) fall between 0 and 1. Since we know the probability of the function, we can then find the odds. For logit functions, the overall graph ranges from -∞ and +∞, since log(0) = -∞ and log(1) = +∞.
# 
# | ![S-LinePredictor](attachment:s_line_predictor_val.png)|
# |:--:|
# |<b>Fig. # - Logit graphs closely resemble an S-shape, and can be reflected across the y-axis due to their perfect symmetry.</b>|
# 
# As you can see above, a logit graph has perfect symmetry and is centered at 0. It closely resembles an S-shape. 
# 
# Overall, the odds are just the ratio of something happening to something not happening. The log of those odds makes things symmetrical and easier to interpret. 

# ## Classifiers
# 
# So, now we know that logistic regression is a form of classification. But how exactly does classification work?
# 
# ![Classification Summary.png](attachment:classification_summary_2.png)
# 
# As you can see from the table above, we determine the “class” of inputs based on the probability of it to be true (also known as _propensity_). One way we can do this is by creating cutoff values; if the probability is greater or equal to the cutoff, then we predict “yes”. If the probability is less than the cutoff, then we predict “no”.
# 
# **How do we determine the cutoff value of X?**
# 
# When answering this question, there are three main things to keep in mind: 
# 
# 
# 1. The incorrect way of predicting class is by setting all classifiers towards the most common prediction. This would set all values to either “yes” or “no” depending on what is most popular.
# 2. The most popular cutoff value is X = 0.5, which serves as the initial choice until a more accurate value is determined. 
# 3. In order to maximize classification accuracy, you should center X according to the set of data at hand. 
# 
# Although we can determine the cutoff value, there are better ways of evaluating the accuracy of the classifier. To be specific, we can use a confusion matrix and/or an ROC curve and then adjust the value of this cutoff. 

# ## Confusion Matrix
# 
# A <span style="color:orange">**confusion matrix** </span> is a table that is used to describe the performance of classification models, or “classifiers.” It is also known as a “classification matrix”. 
# 
# We often see or hear of these measures in the real world, but they are not always easily identifiable as “confusion matrices”. Take, for example, testing the accuracy of new coronavirus detecting tests. 
# 
# | ![Confusion Matrix - General](attachment:confusion_matrix_general.png)|
# |:--:|
# |<b>Fig. # - When measuring the performance of a classification model, these four quartiles help determine the effectiveness of the test. </b>|

# To visualize how a confusion matrix is created, we can create a table with values of the actual outcome and predicted outcome, as determined by regression. After setting our cutoff value at 5, we can then categorize our values to either “Yes” or “No”. 
# 
# ![Actual Predicted Outcome Table](attachment:actual_pred_outcome_table.png)

# | ![Confusion Matrix Values.png](attachment:confusion_matrix_values.png)|
# |:--:|
# |<b>Fig. # - In order to measure classification accuracy, you must look at the matrix from a mathematical perspective.</b>|

# Although the classification matrix itself is pretty easy to understand, sometimes the terminology can be a bit confusing. Below is a reference of all the formulas you’ll need to know:
# 
# 
# 
# *   **N** = # of observation = TP + TN + FP + FN
# *   **Sensitivity =** TP/(TP+FN)
# *   **Specificity =** TN/(TN+FP)
# *   **False Negative Rate =** FN/(TP+FN)
# *   **False Positive Rate =** FP/(TN+FP)
# *   **Overall accuracy** = (TP+TN)/N
# *   **Overall error rate** = (FP+FN)/N
#     *   The sum of overall accuracy AND overall error rate will **always equal 1**. 
# 
# <span style="color:orange">**Sensitivity** </span> is also known as the True Positive Rate, which defines your success of accurately labeling an instance as true. It is also known as “recall." <span style="color:orange">**Specificity** </span> is also known as the True Negative Rate, which defines your success of accurately labeling an instance as false.
# 
# There is a tradeoff between sensitivity and specificity when running these tests. You have to decide which measure is more important based on the case being discussed.

# ## ROC Curve
# 
# Lastly, below we have an ROC Curve, or a Receiver Operating Characteristic Curve. The **ROC curve** graphs sensitivity (the true positive rate) on the y-axis and 1-specificity (the false positive rate) on the x-axis. The curve shows the performance of a classification model at all classification thresholds. Its main goal is to identify as many true positives as possible, while minimizing the amount of false positives. 
# 
# | ![ROC Curve](attachment:roc_curve.png)|
# |:--:|
# |<b>Fig. # -  The closer the ROC curve is to being a perfect classifier, the more accurate it is.</b>|
# 
# By using an ROC curve, we can determine the best cutoff for the best trade-off. The trade-off is measured between False Negative and False Positive Rate. 
# 
# If the cutoff value (t) is large, then the False Positive Rate should be low and the False Negative Rate should be high. In contrast, if the t-value is small, then the False Positive Rate should be high and the False Negative Rate should be low.  
# 
# Let’s take a closer look at the graph above. As you can see, the arrow points upward, indicating that the closer you get to that straight purple line, the better (or more accurate) you will be. You can also measure how accurate the test is by referencing the AUC, or Area Under the Curve. If the AUC is 0.5, then it’s a random classifier, as shown with the red dotted line. If the AUC is 1, then it’s a perfect classifier, as shown in the purple. 
# 
# One last statistic we will look at is the average misclassification cost. This is the calculated cost of incorrectly classifying an observation. In other words, what is the cost of classifying an event as being “No”, when in reality it was “Yes” and vice-versa. In this formula, Q1 is referencing FP (False Positive), while Q2 is referencing FN (False Negative). 
# 
# <span style="color:orange">**Average Misclassification Cost** </span>
# $$
# (Q1*FP + Q2*FN)/N
# $$
# 
