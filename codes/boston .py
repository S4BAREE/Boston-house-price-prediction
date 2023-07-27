#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')
import os
import ydata_profiling as pf
import seaborn as sns 
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split ,GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import pickle


# In[2]:


os.chdir('F:\\boston')


# ##  Lets load the Boston House Pricing Dataset

# In[3]:


df =pd.read_csv('housing.csv')
df


# In[4]:


from sklearn.datasets import load_boston


# In[5]:


boston_df =load_boston()


# In[6]:


def clean_dataset(input_file, output_file, separator=','):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    cleaned_data = []
    for line in lines:
        # Remove leading/trailing whitespaces and split values by spaces
        values = line.strip().split()
        # Join the values using the specified separator
        cleaned_line = separator.join(values)
        cleaned_data.append(cleaned_line)

    with open(output_file, 'w') as f:
        f.write("\n".join(cleaned_data))

# Replace 'dataset.txt' with the path to your input file
input_file_path = 'housing.csv'

# Replace 'cleaned_dataset.txt' with the desired output file path
output_file_path = 'cleaned_dataset.csv'

# Call the function to clean the dataset with commas as separators
clean_dataset(input_file_path, output_file_path, separator=',')


# In[7]:


clean_dataset(input_file_path, output_file_path, separator=';')


# In[8]:


pd.read_csv('cleaned_dataset.csv')


# In[9]:


df =pd.read_csv('housing.csv')
df


# In[10]:


import csv

def clean_csv_dataset(input_file, output_file, delimiter=','):
    with open(input_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')

        cleaned_data = []
        for row in reader:
            # Join the values using the specified delimiter
            cleaned_row = delimiter.join(row)
            cleaned_data.append(cleaned_row)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(cleaned_data)

# Replace 'dataset.csv' with the path to your input CSV file
input_file_path = 'housing.csv'

# Replace 'cleaned_dataset.csv' with the desired output file path
output_file_path = 'cleaned_housing.csv'

# Call the function to clean the CSV dataset with commas as delimiters
clean_csv_dataset(input_file_path, output_file_path, delimiter=',')


# In[11]:


clean_csv_dataset(input_file_path, output_file_path, delimiter=';')


# In[12]:


df =pd.read_csv('cleaned_housing.csv')
df


# In[13]:


df=pd.read_excel('boston price.xlsx')
df


# In[14]:


display(df.columns)


# #### Renaming the column names

# In[15]:


df = df.rename(columns={'Column1':'CRIM','Column2':'ZN', 'Column3':'INDUS','Column4':'CHAS','Column5':'NOX','Column6':'RM',
                      'Column7':'AGE','Column8':'DIS','Column9':'RAD','Column10':'TAX','Column11':'PTRATIO','Column12':'B'
                       ,'Column13':'LSTAT','Column14':'Price'})
display (df.columns)


# In[16]:


display (df)


# In[17]:


df


# In[18]:


print(df.describe)


# In[19]:


df.isna().sum()


# In[20]:


df.duplicated().sum()


# In[21]:


df.shape


# In[22]:


pf.ProfileReport(df)


# ## Preparing Dataset

# In[23]:


df= pd.DataFrame(df)


# In[24]:


df


# In[25]:


df.head()


# In[26]:


df.info()


# ## summerizing the dataset

# In[27]:


df.describe()


# ## Check the missing values

# In[28]:


df.isna().sum()


# ### Exploratory Data Analysis EDA
# #### Correlation

# In[29]:


df.corr()


# In[30]:


plt.rcParams['figure.figsize']=(20,15)
sns.heatmap(df.corr(),annot = True, cmap = 'icefire',square = True,cbar = True)
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# In[31]:


sns.pairplot(df)


# In[32]:


plt.scatter(df['CRIM'],df['Price'])
plt.xlabel('Crime Rate')
plt.ylabel('Price')


# In[33]:


plt.scatter(df['RM'],df['Price'])
plt.xlabel('RM')
plt.ylabel('Price')


# In[34]:


sns.regplot(x='RM',y='Price',data=df)


# In[35]:


sns.regplot(x='LSTAT',y='Price',data=df)


# In[36]:


sns.regplot(x='CHAS',y='Price',data=df)


# In[37]:


sns.regplot(x='PTRATIO',y='Price',data=df)


# #### Independent and Dependent Feature

# In[76]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# In[39]:


x.head()


# In[40]:


y


# ### Train Test Split

# In[41]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(x,y,test_size=0.3,random_state=42)


# In[42]:


xtrain


# In[43]:


xtest


# #### Standardizing the dataset

# In[44]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain=scaler.fit_transform(xtrain)


# In[45]:


xtest = scaler.transform(xtest)


# In[46]:


xtrain


# In[47]:


xtest


# ## Model Training

# In[48]:


from sklearn.linear_model import LinearRegression
lr= LinearRegression()
lr.fit(xtrain,ytrain)


# #### Print the Coefficients and the intercept

# In[49]:


print(lr.coef_)


# In[50]:


print(lr.intercept_)


# In[51]:


lr.get_params()


# #### Prediction with Test data

# In[52]:


ypred = lr.predict(xtest)
ypred


# ### Assumptions
# #### Scatter plot for Prediction

# In[53]:


plt.scatter(ytest,ypred)


# #### Residuals

# In[54]:


residuals = ytest-ypred
residuals


# In[55]:


sns.displot(residuals,kind='kde')


# #### Scatterplot withn respect to prediction and residuals

# In[56]:


plt.scatter(ypred,residuals)
print('Uniform Distribution')


# In[57]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(ytest,ypred))
print(mean_squared_error(ytest,ypred))
print(np.sqrt(mean_squared_error(ytest,ypred)))


# #### R square and Adjusted R square

# In[58]:


from sklearn.metrics import r2_score
score = r2_score(ytest,ypred)
print(score)


# In[59]:


1 - (1-score)*(len(ytest)-1)/(len(ytest)-xtest.shape[1]-1)


# In[82]:


x=df.iloc[:,:-1]
y=df.iloc[:,-1]


# # Model Fitting:
# #### Defining the Function for the ML algorithms using GridSearchCV Algorithm and splitting the dependent variable & independent variable into training and test dataset and Predicting the Dependent Variable by fitting the given model and create the pickle file of the model with the given Algo_name.  Further getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset.

# In[84]:


x


# In[85]:


y


# In[107]:


def FitModel(x,y,algo_name,algorithm,GridSearchParams,cv):
    np.random.seed(10)
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 44)
    grid = GridSearchCV(estimator = algorithm,param_grid = GridSearchParams,cv = cv,
                     scoring = 'r2',verbose = 0, n_jobs = -1)
    grid_result = grid.fit(xtrain,ytrain)
    best_params = grid_result.best_params_
    pred = grid_result.predict(xtest)
    pickle.dump(grid_result,open(algo_name,'wb'))
    print('Algorithm Name:\t',algo_name)
    print('Best Params:',best_params)
    print('R2 Score : {}%'.format(100* r2_score(ytest,pred)))   


# #### Running the function with empty parameters since the Linear Regression model doesn't need any special parameters and fitting the Linear Regression Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Linear Regression

# In[109]:


param = {}
FitModel(x,y,'Linear Regression',LinearRegression(),param,cv=10)


# #### Running the function with empty parameters since the Lasso model doesn't need any special parameters and fitting the Lasso Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Lasso.

# In[110]:


param = {}
FitModel(x,y,'Lasso ',Lasso(),param,cv=10)


# #### Running the function with empty parameters since the Ridge model doesn't need any special parameters and fitting the Ridge Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Ridge.

# In[111]:


param ={}
FitModel(x,y,'Ridge',Ridge(),param,cv=10)


# #### Running the function with some appropriate parameters and fitting the Decision Tree Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Decision Tree.

# In[112]:


"max_features":['auto','sqrt'],
         "max_depth":[int(x) for x in np.linspace(6, 45, num = 5)],
         "min_samples_leaf":[1,2,5,10],
         "min_samples_split":[2, 5, 10, 15, 100],
         "min_weight_fraction_leaf":[0.0,0.1,0.2,0.3,0.4,0.5]


# In[113]:


param= {'max_features':['auto','sqrt'],
       'max_depth':[int(x) for x in np.linspace(6,45,num=5)],
       'min_samples_leaf':[1,2,5,10],
       'min_samples_split':[2,5,10,15,100],
       'min_weight_fraction_leaf':[0.0,0.1,0.2,0.3,0.4,0.5]}

FitModel(x,y,'Decision Tree',DecisionTreeRegressor(),param,cv=10)


# In[114]:


param = { "max_features":['auto','sqrt'],
          "max_depth":[int(x) for x in np.linspace(6, 45, num = 5)],
          "min_samples_leaf":[1,2,5,10],
          "min_samples_split":[2, 5, 10, 15, 100],
          "min_weight_fraction_leaf":[0.0,0.1,0.2,0.3,0.4,0.5]}

FitModel(x,y,'Decision Tree',DecisionTreeRegressor(),param,cv=10)


# In[115]:


param={}
FitModel(x,y,'DecisionTree',DecisionTreeRegressor(),param,cv=10)


# #### Running the function with some appropriate parameters and fitting the Random Forest Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Random Forest.
# 
# 

# In[116]:


param = {'n_estimators':[500,600,800,1000],
         "criterion":["squared_error", "absolute_error", "poisson"]}


# In[117]:


param={}
FitModel(x,y,'Random Forest',RandomForestRegressor(),param,cv=10)


# In[119]:


param ={'n_estimators':[500,600,800,1000],
       'criterion':['squared_error','absolute_error','poisson']}
FitModel(x,y,'Random Forest',RandomForestRegressor(),param,cv=10)


# #### Running the function with some appropriate parameters and fitting the Extra Trees Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name Extra Trees.
# 
# 

# In[120]:


param = { 'n_estimators':[500,600,800,1000],
        'max_features':['auto','sqrt']}
FitModel(x,y,'Extra Trees',ExtraTreesRegressor(),param,cv=10)


# #### Running the function with some appropriate parameters and fitting the XGBoost Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name XGBoost.
# 
# 
# 

# In[124]:


param={'n_estimators':[111,222,333,444]}
FitModel(x,y,'XGBoost',XGBRegressor(),param,cv=10)


# #### Running the function with empty parameters since the CatBoost Regressor model doesn't need any special parameters and fitting the CatBoost Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name CatBoost.
# 
# 

# In[125]:


param={}
FitModel(x,y,'Catboost',CatBoostRegressor(),param,cv=10)


# #### Running the function with empty parameters since the LightGBM Regressor model doesn't need any special parameters and fitting the LightGBM Regressor Algorithm and getting the Algorithm Name, Best Parameters of the algorithm, R2 Score in percentage format between the predicted values and dependent test dataset and also the pickle file with the name LightGBM.

# In[126]:


param={}
FitModel(x,y,'LightGBM',LGBMRegressor(),param,cv=10)


# #### Loading the pickle file with the algorithm which gives highest r2 score percentage
# 
# 

# In[128]:


model=pickle.load(open('CatBoost','rb'))


# #### Predicting the dependent variable using the loaded pickle file and getting the Accuracy Score in percentage format between the predicted values and dependent variable
# 
# 

# In[129]:


pred1=model.predict(x)
print('R2 Score :{}%'.format(100* r2_score(y,pred1)))


# #### Making the Predicted value as a new dataframe and concating it with the original data, so that we can able to compare the differences between Predicted price and Original Price.
# 
# 

# In[130]:


prediction = pd.DataFrame(pred1,columns=['Predicted Price(Approx.)'])
pred_df = pd.concat([df,prediction],axis=1)


# #### Exporting the Data With Prediction of House Price to a csv file
# 
# 

# In[132]:


pred_df.to_csv('Boston  House Price Predicted.csv', index = False)


# #### Plotting the line graph to represent the Accuracy between Predicted Price and Original Price and saving the PNG file
# 
# 

# In[133]:


plt.plot(y,pred1)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.savefig('Actual Price vs Predicted Price.png')
plt.show()


# In[137]:


plt.scatter(y,pred1)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Price')
plt.savefig('Actual Price vs Predicted Price.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




