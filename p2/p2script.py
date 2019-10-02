import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

trainData = pd.read_csv("AdultTrainData.csv", header = 0)
testData = pd.read_csv("AdultTestData.csv", header = 0)

# apply labels to attributes


# transform string values to integer values to check
# correlation

# check attribute correlations
#sns.heatmap(tith arranging headers. For example, we can also have more than one row as header asrainData.corr(), annot=True)
#plt.show()

def transA2(str):
    if str == 'State-gov':
        return 0
    if str == 'Self-emp-not-inc':
        return 1
    if str == 'Private':
        return 2
    if str == 'Federal-gov':
        return 3
    if str == 'Local-gov':
        return 4
    if str == '?':
        return 5
    if str == 'Self-emp-inc':
        return 6
    if str == 'Without-pay':
        return 7
    if str == 'Never-worked':
        return 8

def transA4(str):
    if str == 'Bachelors':
        return 0
    if str == 'HS-grad':
        return 1
    if str == '11th':
        return 2
    if str == 'Masters':
        return 3
    if str == '9th':
        return 4
    if str == 'Some-college':
        return 5
    if str == 'Assoc-acdm':
        return 6
    if str == 'Assoc-voc':
        return 7
    if str == '7th-8th':
        return 8
    if str == 'Doctorate':
        return 9
    if str == 'Prof-school':
        return 10
    if str == '5th-6th':
        return 11
    if str == '10th':
        return 12
    if str == '1st-4th':
        return 13
    if str == 'Preschool':
        return 14
    if str == '12th':
        return 15

def transA6(str):
    if str == 'Never-married':
        return 0
    if str == 'Married-civ-spouse':
        return 1
    if str == 'Divorced':
        return 2
    if str == 'Married-spouse-absent':
        return 3
    if str == 'Separated':
        return 4
    if str == 'Married-AF-spouse':
        return 5
    if str == 'Widowed':
        return 6

def transA7(str):
    if str == 'Adm-clerical':
        return 0
    if str == 'Exec-managerial':
        return 1
    if str == 'Handlers-cleaners':
        return 2
    if str == 'Prof-specialty':
        return 3
    if str == 'Other-service':
        return 4
    if str == 'Sales':
        return 5
    if str == 'Craft-repair':
        return 6
    if str == 'Transport-moving':
        return 7
    if str == 'Farming-fishing':
        return 8
    if str == 'Machine-op-inspct':
        return 9
    if str == 'Tech-support':
        return 10
    if str == '?':
        return 11
    if str == 'Protective-serv':
        return 12
    if str == 'Armed-Forces':
        return 13
    if str == 'Priv-house-serv':
        return 14

def transA8(str):
    if str == 'Not-in-family':
        return 0
    if str == 'Husband':
        return 1
    if str == 'Wife':
        return 2
    if str == 'Own-child':
        return 3
    if str == 'Unmarried':
        return 4
    if str == 'Other-relative':
        return 5

def transA9(str):
    if str == 'White':
        return 0
    if str == 'Black':
        return 1
    if str == 'Asian-Pac-Islander':
        return 2
    if str == 'Amer-Indian-Eskimo':
        return 3
    if str == 'Other':
        return 4

def transA10(str):
    if str == 'Male':
        return 0
    if str == 'Female':
        return 1

def transA14(str):
    if str == 'United-States':
        return 0
    if str == 'Cuba':
        return 1
    if str == 'Jamaica':
        return 2
    if str == 'India':
        return 3
    if str == '?':
        return 4
    if str == 'Mexico':
        return 5
    if str == 'South':
        return 6
    if str == 'Puerto-Rico':
        return 7
    if str == 'Honduras':
        return 8
    if str == 'England':
        return 9
    if str == 'Canada':
        return 10
    if str == 'Germany':
        return 11
    if str == 'Iran':
        return 12
    if str == 'Philippines':
        return 13
    if str == 'Italy':
        return 14
    if str == 'Poland':
        return 15
    if str == 'Columbia':
        return 16
    if str == 'Cambodia':
        return 17
    if str == 'Thailand':
        return 18
    if str == 'Ecuador':
        return 19
    if str == 'Laos':
        return 20
    if str == 'Taiwan':
        return 21
    if str == 'Haiti':
        return 22
    if str == 'Portugal':
        return 23
    if str == 'Dominican-Republic':
        return 24
    if str == 'El-Salvador':
        return 25
    if str == 'France':
        return 26
    if str == 'Guatemala':
        return 27
    if str == 'China':
        return 28
    if str == 'Japan':
        return 29
    if str == 'Yugoslavia':
        return 30
    if str == 'Peru':
        return 31
    if str == 'Outlying-US(Guam-USVI-etc)':
        return 32
    if str == 'Scotland':
        return 33
    if str == 'Trinadad&Tobago':
        return 34
    if str == 'Greece':
        return 35
    if str == 'Nicaragua':
        return 36
    if str == 'Vietnam':
        return 37
    if str == 'Hong':
        return 38
    if str == 'Hungary':
        return 39
    if str == 'Holand-Netherlands':
        return 40

def transTrainClass(str):
    if str == '<=50K':
        return 0
    if str == '>50K':
        return 1

def transTestClass(str):
    if str == '<=50K.':
        return 0
    if str == '>50K.':
        return 1

# apply data transformations
trainData['A2'] = trainData['A2'].apply(transA2)
trainData['A4'] = trainData['A4'].apply(transA4)
trainData['A6'] = trainData['A6'].apply(transA6)
trainData['A7'] = trainData['A7'].apply(transA7)
trainData['A8'] = trainData['A8'].apply(transA8)
trainData['A9'] = trainData['A9'].apply(transA9)
trainData['A10'] = trainData['A10'].apply(transA10)
trainData['A14'] = trainData['A14'].apply(transA14)
trainData['Class'] = trainData['Class'].apply(transTrainClass)

testData['A2'] = testData['A2'].apply(transA2)
testData['A4'] = testData['A4'].apply(transA4)
testData['A6'] = testData['A6'].apply(transA6)
testData['A7'] = testData['A7'].apply(transA7)
testData['A8'] = testData['A8'].apply(transA8)
testData['A9'] = testData['A9'].apply(transA9)
testData['A10'] = testData['A10'].apply(transA10)
testData['A14'] = testData['A14'].apply(transA14)
testData['Class'] = testData['Class'].apply(transTestClass)

# fill NaN values with mean
trainData.fillna(trainData.mean(), inplace = True)
testData.fillna(testData.mean(), inplace = True)

# drop A3 and A9 (weak correlation)
trainData = trainData.drop(['A3','A9'],1)
testData = testData.drop(['A3','A9'],1)

# training instances + labels
X_train = trainData.drop(['Class'],1)
x_labels = trainData['Class']

# test instances + labels
Y_train = testData.drop(['Class'],1)
y_labels = testData['Class']

# KNeighborsClassifier
classifier = KNeighborsClassifier()

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------KNeighbors Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# Naive Bayes
classifier = GaussianNB()

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------Naive Bayes Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# SVM
classifier = SVC(gamma='scale')

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------SVC Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# DecisionTree
classifier = DecisionTreeClassifier()

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------DecisionTree Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# RandomForest
classifier = RandomForestClassifier()

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------RandomForest Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# AdaBoostClassifier
classifier = AdaBoostClassifier()

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------AdaBoost Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# Gradient Boosting
classifier = GradientBoostingClassifier()

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------Gradient Boosting Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# Linear Discriminant Analysis
classifier = LinearDiscriminantAnalysis()

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------LinearDescriminantAnalysis Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# MLP
classifier = MLPClassifier(early_stopping=True, max_iter=800)

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------MLP Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()

# LogisticRegression
classifier = LogisticRegression()

# fit model
classifier.fit(X_train, x_labels)

# predict
y_pred = classifier.predict(Y_train)

print("------------------LogisticRegression Classifier------------------")
print()
df = pd.DataFrame({'Actual': y_labels.values.flatten(), 'Predicted': y_pred.flatten()})
print(df.head())

# calculate scores
accuracy = accuracy_score(y_labels, y_pred)
precision = precision_score(y_labels, y_pred, average='weighted')
recall = recall_score(y_labels, y_pred, average='binary')
f1 = 2 * (precision * recall) / (precision + recall)
auc = metrics.roc_auc_score(y_labels, y_pred)

print()
print("Classification Accuracy: %0.2f" %accuracy)
print("Precision: %0.2f" %precision)
print("Recall: %0.2f" %recall)
print("F1 score: %0.2f" %f1)
print("AUC: %0.2f" %auc)
print()
