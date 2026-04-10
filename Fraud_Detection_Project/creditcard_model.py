import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

df = pd.read_csv("creditcard.csv", low_memory=False)

print(df.head())

print("\nClass distribution:")
print(df['Class'].value_counts())

sns.countplot(x='Class', data=df)
plt.title("Class Distribution")
plt.show()

X = df.drop("Class", axis=1)
Y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, Y_resampled = smote.fit_reasample(X_scaled, Y)

print("\nAfter SMOTE:")
print(pd.Series(Y_resampled).value_counts())

X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, Y_train)

Y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Results")
print(classification_report(Y_test, Y_pred_log))

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, Y_train)

Y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results")
print(classification_report(Y_test, Y_pred_rf))

cm = confusion_matrix(Y_test, Y_pred_rf)

sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()