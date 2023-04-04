# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 10:52:38 2022

@author: Admin
"""

"""Creating a project on Loan dataset -EDA & ML(various algorithm-rfc,logistic regression)"""

# importing libraries

import pandas as pd

# importing data set with the help of pandas (pd.read_csv)

data=pd.read_csv("loan_data_set.csv")

#checking a datast
print(data)

# cheacking our the feature of dataset
data.columns

# now performing EDA on our dataset
# 1) cheacking shape & size,dtypes
 
data.shape
data.dtypes

# checkng nan values present in dataset
data.isnull().sum()

# now filling nan values by "ffill','bfill' method

data.fillna(method="ffill",inplace=True)
data.fillna(method="bfill",inplace=True)

data.isnull().sum()

# Now cheacking duplicates Rows present in our data

data.duplicated().sum()

'''No duplicates present in our data so we  can move ahead'''
# now cheacking head,tail,& distrbution of dataset

data.head()
data.tail()
data.describe()
data.describe(include="all")
data.describe(include="object")
data.std()

# now performing univariate analysis
#checking  approve & o aprove loan status
data["Loan_Status"].value_counts()

#cheacking out in turms of %
data["Loan_Status"].value_counts(normalize=True)*100

# Now plotting a simple bar plot of loan sttus with value_counts
data["Loan_Status"].value_counts().plot(kind="bar")

# checkig outliers having  loanamout with the help of matplotlib

import matplotlib.pyplot as plt

plt.boxplot(data['LoanAmount'])
plt.tittle=("outliers")
plt.ylabel("loan amount")

# now performing bivarite analyais
#cheacking a loan_status based on credit history
pd.crosstab(data["Credit_History"],data['Loan_Status'])

#now cheacking correlaton 
x=data.corr()
#adding a new feature in data i.e "Applicant_in_INR" & then find any corr with help of heat map

data["Applicant_in_INR"]=data["ApplicantIncome"]*83

import seaborn as sns
y=data.corr()

sns.heatmap(y)
#creating a graph for average lone amount sansation with respect to dataset with the group function
data.groupby("Loan_Status").agg({'LoanAmount':'mean'})
 

# now we start to apply ML model
# Randomforest classifier (rfc)

import category_encoders as ce
#choose x & y variable ie input &target with the help of iloc

x=data.iloc[:,5:11]
y=data.iloc[:,-2]

# now splitting in train & test dataset with the help of sklearn

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=(40))

# feature transforming
x.columns

encoder=ce.OrdinalEncoder(['Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History'])
      
x_train=encoder.fit_transform(x_train)
x_test=encoder.transform(x_test)

# now calling a model & fitting data in it

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=100,random_state=(40))

rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)

# now finding accuracy for prediction with te help of sklearn

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,y_pred))

# finding feature score(feature importance) 

frature_score= pd.Series(rfc.feature_importances_,index=x_train.columns)

print(frature_score)

sns.barplot(frature_score,x_train.columns,)
plt.xlabel("accuracy_score")
plt.ylabel("feature")
plt.tittle("feature_score & accuracy")

# as we can see the feature self_employed having very less featue score then we con remove it 

#now genretimg classification report

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# now applying logistic regression
# importing data & libraries
import pandas as pd
df=pd.read_csv("C:\\Users\\Admin\\Desktop\\loan_data_set.csv")

#finding & filling null or missing values
df.isnull().sum()

df.fillna(method="ffill",inplace=True)
df.fillna(method="bfill",inplace=True)
df.isnull().sum()

# appying one ot encoding
df1=pd.get_dummies(df,columns=["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status"])
df1.dtypes

#dropping Extra  dummies columns
df1.drop(["Gender_Female","Married_No","Education_Graduate","Self_Employed_No","Property_Area_Rural","Loan_Status_N"],axis=1,inplace=True)
# deviding input & target variable
x=df1.iloc[:,2:13]
y=df.iloc[:,-1]
 
# splitting data into train & test with help of sklearn
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30)

# applying model with the help of sklearn
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

# finding accuracy of our mpdel
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc

# our model preditctiong with 82% accuracy

