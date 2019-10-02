import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor

from sklearn import metrics
import csv

data = pd.read_csv("diamonds.csv")

# drop index column
data = data.drop(data.columns[0], 1)

# set string cut values to integers
def cutInt(v):
    if v =="Ideal": return 2
    if v =="Premium": return 1
    if v =="Good": return 4
    if v =="Very Good": return 3
    if v =="Fair": return 5

# set string color values to integers
def colorInt(v):
    if v =="E": return 2
    if v =="I": return 1
    if v =="J": return 4
    if v =="H": return 3
    if v =="F": return 5
    if v =="G": return 6
    if v =="D": return 7

# set string clarity values to integers
def clarityInt(v):
    if v =="SI2": return 2
    if v =="SI1": return 1
    if v =="VS1": return 4
    if v =="VS2": return 3
    if v =="VVS2": return 5
    if v =="VVS1": return 6
    if v =="I1": return 7
    if v =="IF": return 8

# apply string to int transformations
data['cut'] = data['cut'].apply(cutInt)
data['color'] = data['color'].apply(colorInt)
data['clarity'] = data['clarity'].apply(clarityInt)

#print('-----------Peak at head of the data-----------')
#print(data.head())

# split to attributes and labels
X = data.drop(['price'], 1)
y = data['price']

# create train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=309)

# store current time
startTime = datetime.datetime.now()

# train model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

# store end time
endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print('-----------LINEAR REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#K-neighbours regression
startTime = datetime.datetime.now()

regressor = KNeighborsRegressor(weights='distance')
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print()
print('-----------K-NEIGHBOURS REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#Ridge regression
startTime = datetime.datetime.now()

regressor = Ridge(alpha=1.0)
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print()
print('-----------RIDGE REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#Decision tree regression
startTime = datetime.datetime.now()

regressor = DecisionTreeRegressor()

regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print()
print('-----------DECISION TREE REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#Random forest regression
startTime = datetime.datetime.now()

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print()
print('-----------RANDOM FOREST REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#Gradient boosting regression
startTime = datetime.datetime.now()

regressor = GradientBoostingRegressor(n_iter_no_change=500, validation_fraction=0.3)
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print()
print('-----------GRADIENT BOOSTING REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#SGD regression
startTime = datetime.datetime.now()

regressor = SGDRegressor()
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print()
print('-----------SGD REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#Support vector regression
startTime = datetime.datetime.now()

regressor = SVR(gamma='scale', C=1.0, epsilon=0.2)
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print()
print('-----------SVR REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#Linear SVM regression
startTime = datetime.datetime.now()

regressor = LinearSVR()
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()

# outpt results actual vs predicted
print()
print('-----------LINEAR SVR REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)

#MLP regression
startTime = datetime.datetime.now()

regressor = MLPRegressor(early_stopping=True, max_iter=800)
regressor.fit(X_train, y_train)

# predict
y_pred = regressor.predict(X_test)

endTime = datetime.datetime.now()
executionTime = (endTime - startTime).total_seconds()


# outpt results actual vs predicted
print()
print('-----------MLP REGRESSION-----------')
df = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print()
print('-----------Actual vs Predicted-----------')
#peak first 5 instances
print(df.head())


# evaluate performance
print()
print('-----------Error values-----------')
print('Mean Absolute Error: %0.2f ' %metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error: %0.2f ' %metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error: %0.2f ' %np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-squared Error: %0.2f ' %metrics.r2_score(y_test, y_pred))
print('Execution Time: %0.2f' %executionTime)
