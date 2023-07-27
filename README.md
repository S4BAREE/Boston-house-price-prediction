# Boston-house-price-prediction


 
# Overview:
This application is designed to predict house prices in Boston using a machine learning model. It utilizes historical data on various housing features in the Boston area to make accurate predictions. The model is built using the scikit-learn library in Python, and it employs the popular Boston Housing dataset for training and evaluation.



# Problem Statement:
The problem addressed in the Boston House Price Prediction project is the difficulty faced byreal estate agents, potential homebuyers and investors in accurately determining the prices of houses in the Boston area. The housing market is influenced by various factors, and understanding the relationship between these factors and house prices can be complex and time-consuming. The lack of a reliable and accurate prediction model hinders decision-making processes and can result in suboptimal investments or missed opportunities.The goal of this project is to develop a data-driven solution that can predict housing prices in Boston with a high degree of accuracy.


# Dataset:
The Boston Housing dataset used in this application contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. The dataset contains 506 data points, with each data point having 14 features. The target variable is the median value of owner-occupied homes in $1000s.
# Model Training:
The application utilizes a linear regression model to predict house prices. The model is trained on the Boston Housing dataset, and 80% of the data is used for training, while 20% is used for testing the model's accuracy.

# Observations:
The project found that the following factors are most important in prediction the price of house in Boston.
CRIM - Per Capita Crime Rate by town
ZN - Proportion of Residential Land Zoned for lots over 25,000 sq.ft.
INDUS - Proportion of Non-retail Business acres per town
CHAS - Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
NOX - Nitric Oxides Concentration (parts per 10 million)
RM - Average Number of Rooms per dwelling
AGE - Proportion of Owner-occupied units built prior to 1940
DIS - Weighted Distances to Five Boston Employment Centres
RAD - Index of Accessibility to Radial Highways
TAX - Full-value Property-tax rate per 10,000 dollars
PTRATIO - Pupil-Teacher Ratio by town\
B - "1000(Bk - 0.63)^2" where Bk is the proportion of Blacks by town
LSTAT - Percentage Lower Status of the Population

# Findings:
The project found that the machine learning model was able to predict the price of a house in Boston with a high degree of accuracy. The model was able to predict the price of a house in Boston within 96% of the accuracy. The project also found that the model was able to generalize well to new data. The model was able to predict the price of house in Boston if all the required data is given.

# License:
The Boston House Price Prediction application is released under the [MIT License](https://opensource.org/licenses/MIT). You are free to modify and distribute the code as per the terms of the license.
# Conclusion:
Achieved in developing a predictive model to predict theprice of houses in Boston city based on given data with accuracy of 96.14% (CatBoost Regressor)
