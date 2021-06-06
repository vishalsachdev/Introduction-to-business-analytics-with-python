#!/usr/bin/env python
# coding: utf-8

# # EDA + Data Visualization

# Exploratory data analysis is comprised of three parts:

# **Data Inspection:**
# 
# This will start by looking at the raw data. This can be done by using functions such as head(), describe(), info(), etc.
# 
# The head() function will print the first n rows of the dataset.
# 
# ```
# random_df.head(5)
# ```
# 
# The tail() function is similar to the head() function; it will print the last n rows of the dataset.
# 
# ```
# random_df.tail(5)
# ```
# 
# The describe() function computes summary statistics of your data, i.e. mean, count, median, quartile 1 value, quartile 3 value, and more.
# 
# ```
# random_df.describe()
# ```
# 
# The info() function prints a summary of your dataset. Specifically, it identifies the number of non-null observations per variable as well as the type of each variable.
# 
# ```
# random_df.info()
# ```

# **Data Exploration:**
# 
# Data exploration calls for exploring further relationships between variables. Examples of ways to explore these relationships are by cross-tabulation (which is used for analysis of categorical data) and dimension reduction (which is used to reduce the amount of unnecessary data in your model), both of which improve accuracy and speed up computing. 
# 
# **Data Visualization**
# 
# Data visualization calls for visualizing relationships between variables. Some of the common data visualizations are as follows:

# ## Basic Charts
# 
# These charts will display two variables (or columns) of data.
# 
# Ex: Bar charts, scatter plots, line graphs

# *   **Bar Charts** - compare a **single statistic** across groups
#     *   Horizontal (x-axis): categorical variable
#     *   Vertical (y-axis): “statistic” of a numerical variable (ex. count, mean, median, etc.)
#   <tr>
#    <td>
# 
# 
# ```
# ax = random_df.groupby('categorical_variable_of_interest').categorical_variable_of_interest.mean().plot
# (kind='bar', figsize=[10, 6], color='orangered')
# 
# ax.set_ylabel('label_here')
# 
# plt.tight_layout()
# plt.show()
# ```
# **(Bar chart example needed here)**

# *   **Scatter plot** - comparing two **numerical variables**
#     *   Horizontal (x-axis): numerical variable
#     *   Vertical (y-axis): also a numerical variable
#     
# ```
# random_df.plot.scatter(x= "variable_1", y="variable_2", legend=False, color='orange')
# ```
# **(Scatter plot example needed here)**

# *   **Line graph** - display **numerical variable** (or statistic) over **time**
#     *   Horizontal (x-axis): date/time variable
#     *   Vertical (y-axis): numerical variable
# 
# First, add Date/Time variable
# ```
# random_df['Date'] = pd.to_datetime(random_df.dteday, format='%m/%d/%Y')
# ```
# Next, set Date column as index
# ```
# random_ts = pd.Series(random_df.cnt.to_numpy(), index=random_df["Date"])
# ```
# **Note: This is a time series, not a data frame.**
# 
# Lastly, plot your Line Graph.
# ```
# random_ts.plot(ylim=[0, 10000], 
#                legend=False, 
#                figsize=[6, 4], 
#                color='darkorange')
# 
# plt.xlabel('x_label_here')  # set x-axis label
# plt.ylabel('y_label_here')  # set y-axis label
# 
# plt.tight_layout()
# plt.show()
# ```
# **(Line graph example needed here)**

# **Distribution Plots**
# 
# Display distribution of a single numerical variable
# 
# Ex. Histograms, Box plots

# *   **Histogram** - represents **frequency** of values with vertical bars
#     *   Horizontal (x-axis): numerical variable, potentially “binned”
#     *   Vertical (y-axis): frequency for that same variable
#     *   Note: For continuous data or data with many possible values, **group values into bins** or a series of ranges (ex. One bin contains all values between 20 and 40)
# 
# ```
# fig, ax = plt.subplots(1,1)
# 
# sns.histplot(random_df.numerical_variable_of_interest, bins=10, color='orange')
# 
# ax.set_title('Variable Distribution', fontsize=20)
# ax.set(xlabel='x_label', ylabel='count')
# ```
# **(Histogram Example needed here)**

# *   **Box plots** - Display **range** and partial distribution of a numerical value
#     *   Quartile 1: 25% of the data below/to the left of this point
#     *   Median: 50% of the data below/to the left of this point; “Middle” value of a variable
#     *   Quartile 3: 75% of the data below/to the left of this point
#     *   Interquartile Range (IQR): IQR = Q3 - Q1; measure of dispersion
# ```
# ax = data_df.boxplot(column='numerical_variable')
# ax.set_ylabel('y_label')
# plt.suptitle('') # Suppress the overall title
# plt.title('')
# plt.show()
# ```
# **(Basic boxplot example here)**

# *   Box plot for comparison - Often used to compare distribution across groups (against categorical variables)
#     *   Horizontal (x-axis): categorical variable
#     *   Vertical (y-axis): numerical variable
# ```
# ax = random_df.boxplot(column='numerical_variable', by='categorical_variable')
# ax.set_ylabel('y_label')
# plt.suptitle('')  # Suppress the overall title
# plt.title('')
# plt.show()
# ```
# **(Example of comparison boxplot here)**

# ## Special charts
# 
# Correlation tables are a (relatively) concise way to display the correlation between different variables in our dataframe. However, they show a lot of information and can be difficult to digest. We solve this by using a heat map.
# 
# **Correlation table:**
# 
# ```
# random_df.corr()
# ```
# **(Example of correlation table here)**
# 
# **Correlation heatmap:**
# 
# Below, we use the correlation table defined above, but we add a heatmap. Let's breakdown the code:
# 
# *   sns: using the seaborn library
# *   heatmap: call heatmap function
#     *   corr: first positional argument, the correlation table
#     *   annot: if True, write data values in each cell
#     *   fmt: string formatting code for each value, # indicates decimal places
#     *   cmap: colormap to use, RdBu is another
# 
# ```
# corr = random_df.corr() # create correlation table
# fig, ax = plt.subplots()
# fig.set_size_inches(11, 7)
# sns.heatmap(corr, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
# plt.show()
# ```
# **(Example of heatmap here)**
# 
# **Missing Value Analysis:**
# 
# Finally, we can generate a bar chart to quickly see which variables are missing values. 
# 
# ```
# msno.bar(random_df, color='deepskyblue')
# ```
# 
# **(Example of graph here)**
