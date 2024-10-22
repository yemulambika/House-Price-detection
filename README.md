# House-Price-detection
By Using machine learning detected the price of House
# Step : 1 Import Libraries
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing 
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline 
pd.set_option('display.max_columns', None)
# Step : 2 Loading The Dataset To A Pandas DataFrame

### Train Data & Test Data
dataset_Train=pd.read_csv('train house.csv')
dataset_Test=pd.read_csv('test house.csv')
### First 5 Rows Of The Train Dataset
dataset_Train.head()
### First 5 Rows OF The Test Dataset
dataset_Test.head()
### Find The Shape Of Train Dataset
dataset_Train.shape
### Find The Shape Of Test Dataset
dataset_Test.shape
# Step : 3 Exploratory Data Analysis
### Checking The Columns In Test Dataset
dataset_Test.columns
### Checking The Columns In Train Dataset
dataset_Train.columns
### Checking The Shape Of Train & Test Dataset
print(dataset_Train.shape, dataset_Test.shape)
### Find The Datatype Of Each Columns In Train Dataset
dataset_Train.info()
### Find The Datatype Of Each Columns in Test Dataset
dataset_Test.info()
pd.set_option('max_rows',None)
### Find The Total Number Of Missing Values In Each Column Train Dataset
dataset_Train.isnull().sum()
dataset_Train.isnull().sum().sum()
### Find The Total Number Of Missing Values In Each Column Test Dataset
dataset_Test.isnull().sum()
dataset_Test.isnull().sum().sum()
### Drop a columns In Train Dataset ( Id,Alley,FireplaceQu,PoolQc,Fence,MiscFeature)
dataset_Train=dataset_Train.drop(columns=['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'])
dataset_Train.shape
dataset_Train.isnull().sum().sum()
### Drop a columns In Test Dataset ( Id,Alley,FireplaceQu,PoolQc,Fence,MiscFeature)
dataset_Test=dataset_Test.drop(columns=['Id','Alley','FireplaceQu','PoolQC','Fence','MiscFeature'])
dataset_Test.shape
dataset_Test.isnull().sum().sum()
### Use LableEncoder Function Convert to all Text data Randomly Into Any Numeric Numbers
### Train Dataset
encoder = preprocessing.LabelEncoder()

for i in dataset_Train.columns:
    if isinstance(dataset_Train[i][0], str):
            dataset_Train[i] = encoder.fit_transform(dataset_Train[i])
dataset_Train.head()
dataset_Train.describe()
### Use LableEncoder Function Convert to all Text data Randomly Into Any Numeric Numbers
### Test Dataset
encoder = preprocessing.LabelEncoder()

for i in dataset_Test.columns:
    if isinstance(dataset_Test[i][0], str):
            dataset_Test[i] = encoder.fit_transform(dataset_Test[i])
dataset_Test.head()
dataset_Test.describe()
dataset_Test.isnull().sum().sum()
dataset_Train.isnull().sum().sum()
## Replacing Missing Values In Train Dataset
dataset_Train['LotFrontage'].fillna(dataset_Train['LotFrontage'].median(), inplace=True)
dataset_Train['MasVnrArea'].fillna(dataset_Train['MasVnrArea'].median(), inplace=True)
dataset_Train['GarageYrBlt'].fillna(dataset_Train['GarageYrBlt'].median(), inplace=True)
dataset_Train.isnull().sum().sum()
## Replacing Missing Values In Test Dataset
dataset_Test['LotFrontage'].fillna(dataset_Test['LotFrontage'].median(), inplace=True)
dataset_Test['MasVnrArea'].fillna(dataset_Test['MasVnrArea'].median(), inplace=True)
dataset_Test['GarageYrBlt'].fillna(dataset_Test['GarageYrBlt'].median(), inplace=True)
dataset_Test['BsmtFullBath'].fillna(dataset_Test['BsmtFullBath'].median(), inplace=True)
dataset_Test['BsmtHalfBath'].fillna(dataset_Test['BsmtHalfBath'].median(), inplace=True)
dataset_Test['BsmtFinSF1'].fillna(dataset_Test['BsmtFinSF1'].median(), inplace=True)
dataset_Test['BsmtFinSF2'].fillna(dataset_Test['BsmtFinSF2'].median(), inplace=True)
dataset_Test['BsmtUnfSF'].fillna(dataset_Test['BsmtUnfSF'].median(), inplace=True)
dataset_Test['TotalBsmtSF'].fillna(dataset_Test['TotalBsmtSF'].median(), inplace=True)
dataset_Test['GarageCars'].fillna(dataset_Test['GarageCars'].median(), inplace=True)
dataset_Test['GarageArea'].fillna(dataset_Test['GarageArea'].median(), inplace=True)
dataset_Test.isnull().sum().sum()
print(dataset_Train.shape, dataset_Test.shape)
dataset_Train.info()
dataset_Test.info()
## Check For Duplicates Values In Train Dataset
duplicate = dataset_Train[dataset_Train.duplicated()]
duplicate
## Check For Duplicates Values In Test Dataset
duplicate = dataset_Test[dataset_Test.duplicated()]
duplicate
## Create A Histogram Of All The Attributes In  Train Dataset
dataset_Train.hist(bins=5, figsize=(21,25))
plt.show()
## Create A Histogram Of All The Attributes In Test Dataset
dataset_Test.hist(bins=5, figsize=(35,31))
plt.show()
## Corelation Of Columns With One Another In Train dataset
corelation=dataset_Train.corr()
corelation
## Corelation Of Columns With One Another In Test dataset
corelation=dataset_Test.corr()
corelation
## Create a Heatmap Of Correlation Of Columns With One Another In Train Dataset
plt.figure(figsize=(30,25))
sns.heatmap(dataset_Train.corr(), annot=True)
## Create a Heatmap Of Correlation Of Columns With One Another In Test Dataset
plt.figure(figsize=(15,30))
sns.heatmap(dataset_Test.corr(), annot=True)
## Correlation Of All Columns With Target Variable. More The Value Is Away From Zero, More The Feature Importance
dataset_Train.corr()['SalePrice'].sort_values(ascending=False)
dataset_Train['SalePrice'].describe()
## Create A Distplot with SalePrice Column
plt.figure(figsize=(15,10))
sns.distplot(dataset_Train['SalePrice'],kde=True);

## Create A BoxPlot & Scatter Plot SalePrice With OverallQual
sns.boxplot(y='SalePrice', x = 'OverallQual', data=dataset_Train)
plt.scatter(y='SalePrice', x = 'OverallQual', data=dataset_Train)
## Outlier Treatment In LotArea Column
plt.scatter(y='SalePrice', x = 'LotArea', data=dataset_Train)
dataset_Train = dataset_Train.drop(dataset_Train[dataset_Train['LotArea'] > 100000].index)
plt.scatter(y='SalePrice', x = 'LotArea', data=dataset_Train)
## Outlier Treatment In LotConfig Column
plt.scatter(y='SalePrice', x = 'LotConfig', data=dataset_Train)
dataset_Train = dataset_Train.drop(dataset_Train[dataset_Train['LotConfig'] > 650000].index)
plt.scatter(y='SalePrice', x = 'LotConfig', data=dataset_Train)
plt.scatter(dataset_Train.index, dataset_Train['Neighborhood'])
## OutlierTreatment In MasVnrArea Column
plt.scatter(dataset_Train.index, dataset_Train['MasVnrArea'])
li = list(dataset_Train['MasVnrArea'].sort_values()[-2:].index)
dataset_Train['MasVnrArea'][li] = int(dataset_Train.drop(li)['MasVnrArea'].mean())
plt.scatter(dataset_Train.index, dataset_Train['MasVnrArea'])
##  Outlier Treatment in TotalBsmtSf Column
plt.scatter(dataset_Train.index, dataset_Train['TotalBsmtSF'])
li = list(dataset_Train['TotalBsmtSF'].sort_values()[-5:].index)
dataset_Train['TotalBsmtSF'][li] = int(dataset_Train.drop(li)['TotalBsmtSF'].mean())
plt.scatter(dataset_Train.index, dataset_Train['TotalBsmtSF'])
## Oultlier Treatment In GrLiveArea Column
plt.scatter(dataset_Train.index, dataset_Train['GrLivArea'])
li = list(dataset_Train['GrLivArea'].sort_values()[-4:].index)
dataset_Train['GrLivArea'][li] = int(dataset_Train.drop(li)['GrLivArea'].mean())
plt.scatter(dataset_Train.index, dataset_Train['GrLivArea'])
## Oultlier Treatment In PoolArea Column
plt.scatter(dataset_Train.index, dataset_Train['PoolArea'])
li = list(dataset_Train['PoolArea'].sort_values()[-7:].index)
dataset_Train['PoolArea'][li] = int(dataset_Train.drop(li)['PoolArea'].mean())
plt.scatter(dataset_Train.index, dataset_Train['PoolArea'])
plt.scatter(dataset_Train.index, dataset_Train['GarageYrBlt'])
##  Oultlier Treatment In MiscVal Column  
plt.scatter(dataset_Train.index, dataset_Train['MiscVal'])
li = list(dataset_Train['MiscVal'].sort_values()[-2:].index)
dataset_Train['MiscVal'][li] = int(dataset_Train.drop(li)['MiscVal'].mean())
plt.scatter(dataset_Train.index, dataset_Train['MiscVal'])
## Add A Training & Testind data
house_data = dataset_Train.append(dataset_Test)
house_data.shape
house_data["SaleType"].fillna(house_data["SaleType"].mode()[0], inplace = True)
house_data["SalePrice"].fillna(house_data["SalePrice"].mode()[0], inplace = True)
house_data=house_data.astype({'TotalBsmtSF':'int','MasVnrArea':'int','LotFrontage':'int','BsmtFinSF1':'int','BsmtFinSF2':'int','BsmtFullBath':'int','BsmtHalfBath':'int','BsmtUnfSF':'int','GarageArea':'int','GarageCars':'int','GarageYrBlt':'int'})


house_data.info()
house_data=house_data.astype({'SalePrice':'int'})
house_data.info()
x = house_data.drop(columns=['SalePrice'])
y = house_data['SalePrice']
## Now We Will Try ML Algorithmns
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, random_state=42)
## Model Linear Reg
from sklearn import linear_model
import pandas as pd
lr = linear_model.LinearRegression()
lr.fit(x,y)
lr.intercept_
lr.coef_
y_pred=lr.predict(x_test)
#pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
lr.score(x,y)
from sklearn.metrics import r2_score,mean_squared_error
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
print('R squre=',r2)
print('Mean Square Error=',mse)
n=x_test.shape[0]
k=x_test.shape[1]
adj_r2=1-(((n-1)/(n-k-1))*(1-r2))
print('Adjusted R2=',adj_r2)
# Model Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dtc = DecisionTreeClassifier(criterion = 'entropy',max_depth = 10,max_features = 'auto')
dtc.fit(x_train, y_train)
pred = dtc.predict(x_test)
dtc.score(x,y)
# Model SVM 
from sklearn import svm
sv = svm.SVC(kernel = 'rbf', C= 1, gamma = 'auto')
sv.kernel
sv.fit(x_train,y_train)
sv.score(x_train,y_train)
# Model Random Forest Reg
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42)
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
