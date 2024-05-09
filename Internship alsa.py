import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("E:\My programming\customer_train.csv")

print(data.head())

print(data.isnull().sum())

print(data.describe())

print(data.dtypes)

plt.figure(figsize=(10, 6))
sns.histplot(data['Year_Birth'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Year of Birth')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(data['Income'])
plt.title('Boxplot of Income')
plt.xlabel('Income')
plt.show()

data.fillna(method='ffill', inplace=True)  # Forward fill missing values

data = pd.get_dummies(data, columns=['Education', 'Marital_Status'])

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'], format='%d-%m-%Y')

data['Year_Customer'] = data['Dt_Customer'].dt.year
data['Month_Customer'] = data['Dt_Customer'].dt.month

data.drop('Dt_Customer', axis=1, inplace=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data[['Income', 'Year_Birth', 'Recency']] = scaler.fit_transform(data[['Income', 'Year_Birth', 'Recency']])

X = data.drop(['ID', 'Complain'], axis=1)
y = data['Complain']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building and evaluating a Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=1))
