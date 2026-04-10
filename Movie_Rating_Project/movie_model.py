import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("movie.csv", encoding="latin1")

print(df.head())

df = df[['Genre',  'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Year', 'Duration', 'Rating']]

df = df.dropna()

df['Year'] = df['Year'].str.replace('(','').str.replace(')','').astype(int)

df['Duration'] = df['Duration'].str.replace(' min','').astype(int)

le = LabelEncoder()

df['Genre'] = le.fit_transform(df['Genre'])
df['Director'] = le.fit_transform(df['Director'])
df['Actor 1'] = le.fit_transform(df['Actor 1'])
df['Actor 2'] = le.fit_transform(df['Actor 2'])
df['Actor 3'] = le.fit_transform(df['Actor 3'])

X = df[['Genre',  'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Year', 'Duration']]
Y = df['Rating']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Mean Squared Error:", mse)
print("R2  Score:", r2)

new_movie = [[1, 5, 10, 15, 25, 2026, 120]]
predicted_rating = model.predict(new_movie)

print("Predicted Ratring:", predicted_rating)
print(df.columns)