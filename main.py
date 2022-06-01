from tkinter import *
import numpy as np   # for numerical operations
import pandas as pd  # for handling input data
import matplotlib.pyplot as plt  # for data visualization
import seaborn as sns  # for data visualization
import statistics
import math
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from pandas.core.common import random_state
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix


def DataInfo(train):
    print(train.info())
    print(train.isnull().sum())
    print(train.head())
    print("------------------------------------------\n\n")


train = pd.read_csv('loan_data.csv')
print(f"Data has {train.shape[0]} Rows and {train.shape[1]} Features")
DataInfo(train)


"""Drop Duplicates"""
# train = train.drop_duplicates


"""Encoding data"""
train= train.apply(LabelEncoder().fit_transform)
DataInfo(train)


"""Remove Null Values"""
# train["Loan_Amount_Term"]=train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mean())
# train["Dependents"]=train["Dependents"].fillna(train["Dependents"].mean())
# train["Dependents"]=train["Dependents"].fillna(train["Credit_History"].mode())
# train["Loan_Amount_Term"]=train["Loan_Amount_Term"].fillna(train["Loan_Amount_Term"].mean())
# train["LoanAmount"]=train["LoanAmount"].fillna(train["LoanAmount"].mean())
# train["Self_Employed"]=train["Self_Employed"].fillna(train["Self_Employed"].mean())
# train["Married"]=train["Married"].fillna(0)
# train["Credit_History"]=train["Credit_History"].fillna(train["Credit_History"].mean())
# train["Gender"]=train["Gender"].fillna(1)

train=train.drop(columns=["Loan_ID"])
DataInfo(train)


"""**Outliers Removal**"""
def RemoveOutliers(ColumnName, train):
    train[ColumnName] = sorted(train[ColumnName])
    Q1, Q3 = np.percentile(train[ColumnName], [25, 75])
    IQR = Q3 - Q1
    outright = (1.5 * IQR) + Q3
    outleft = Q1 - (1.5 * IQR)
    mask1 = train[ColumnName] < outright
    mask2 = train[ColumnName] > outleft
    train = train[mask1]
    train = train[mask2]
    # print(mask1_4.value_counts())

RemoveOutliers('ApplicantIncome', train)
RemoveOutliers('CoapplicantIncome', train)
RemoveOutliers('LoanAmount', train)
RemoveOutliers('Property_Area', train)
DataInfo(train)


"""**Normalization**"""
train = (train - train.min())/ (train.max() - train.min())
DataInfo(train)


"""**Algorithms**"""
y=train["Loan_Status"]
x=train
x=x.drop(columns=["Loan_Status"])
xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.3, random_state=300)


"""*Form*"""
x_dimensional = 800
y_dimensional = 650
frm = Tk()
frm.title("Loan_Prediction")
frm.geometry(str(x_dimensional)+"x"+str(y_dimensional))
Label(frm, text="Loan Prediction", font=("Kartika", 45, "underline"), fg='#0000cc').place(x=200, y=10)
title_color="#314f81"


"""** SVM **"""
Label(frm, text="SVM :- ", font=("Arial", 15), fg=title_color).place(x=10, y=140)
support=svm.SVC(kernel='linear', C=1)
support.fit(xtrain, ytrain)
yprd=support.predict(xtest)
Label(frm, text="Accuracy : "+str(metrics.accuracy_score(ytest, yprd))).place(x=20, y=180)
Label(frm, text="Precision : "+str(metrics.precision_score(ytest, yprd))).place(x=20, y=205)
Label(frm, text="Recall : "+str(metrics.recall_score(ytest, yprd))).place(x=20, y=230)


"""** Logistic Regression **"""
Label(frm, text="Logistic Regression :- ", font=("Arial", 15), fg=title_color).place(x=10, y=305)
model = LogisticRegression(solver='liblinear', C=10, random_state=0)
model.fit(xtrain, ytrain)
yprd = model.predict(xtest)
Label(frm, text="Accuracy : "+str(metrics.accuracy_score(ytest, yprd))).place(x=20, y=345)
Label(frm, text="Precision : "+str(metrics.precision_score(ytest, yprd))).place(x=20, y=370)
Label(frm, text="Recall : "+str(metrics.recall_score(ytest, yprd))).place(x=20, y=395)
# Label(frm, text=str(confusion_matrix(ytest, yprd))).place(x=200, y=395)


"""** ID3 **"""
Label(frm, text="ID3 :- ", font=("Arial", 15), fg=title_color).place(x=10, y=470)
dt=tree.DecisionTreeClassifier(max_depth=2)
dt.fit(xtrain, ytrain)
yprd = dt.predict(xtest)
Label(frm, text="Accuracy : "+str(metrics.accuracy_score(ytest, yprd))).place(x=20, y=510)
Label(frm, text="Precision : "+str(metrics.precision_score(ytest, yprd))).place(x=20, y=535)
Label(frm, text="Recall : "+str(metrics.recall_score(ytest, yprd))).place(x=20, y=560)


"""** KNN **"""
Label(frm, text="KNN :- ", font=("Arial", 15), fg=title_color).place(x=460, y=140)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(xtrain, ytrain)
yprd = knn.predict(xtest)
Label(frm, text="Accuracy : "+str(metrics.accuracy_score(ytest, yprd))).place(x=470, y=180)
Label(frm, text="Precision : "+str(metrics.precision_score(ytest, yprd))).place(x=470, y=205)
Label(frm, text="Recall : "+str(metrics.recall_score(ytest, yprd))).place(x=470, y=230)


"""** Random Forest **"""
Label(frm, text="Random Forest Classifier :- ", font=("Arial", 15), fg=title_color).place(x=460, y=305)
dt1=RandomForestClassifier(max_depth=2, n_estimators=6)
dt1.fit(xtrain, ytrain)
yprd = dt1.predict(xtest)
Label(frm, text="Accuracy : "+str(metrics.accuracy_score(ytest, yprd))).place(x=470, y=345)
Label(frm, text="Precision : "+str(metrics.precision_score(ytest, yprd))).place(x=470, y=370)
Label(frm, text="Recall : "+str(metrics.recall_score(ytest, yprd))).place(x=470, y=395)



Button(frm, text="Exit", command=frm.destroy, font=("Arial", 13), width=8, height=1, fg="#b30000").place(x=x_dimensional-120, y=y_dimensional-65)
frm.mainloop()