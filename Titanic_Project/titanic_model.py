import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as  sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import  LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

df = pd.read_csv("titanic.csv")
print(df.head())

print(df.info())
print(df.describe())
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
                                       
df = df.drop(columns=['Cabin'])

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

df.drop(['Name','Ticket','PassengerId'], axis=1, inplace=True)

X = df.drop('Survived', axis=1)
Y = df['Survived']

X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

new_passenger = [[3, 0, 22, 1, 0, 7.25, 2]]

prediction = model.predict(new_passenger)

if prediction[0] == 1:
    print("Passenger Survived")
else:
    print("Passenger did not Survived")

sns.countplot(x='Survived', data=df)
plt.show()

sns.countplot(x='Sex', hue='Survived', data=df)
plt.show()