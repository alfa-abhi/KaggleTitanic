from sklearn import tree
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd

df_train = pd.read_csv("train2.csv", na_values=[""])
df_train.isnull().any()
df_train = df_train.fillna(method='ffill')
print df_train

y_train = df_train.iloc[0:, 0:1].values
X_train = df_train.iloc[0:, 1:].values


df_test = pd.read_csv("test2.csv", na_values=[""])
df_test.isnull().any()
df_test = df_test.fillna(method='ffill')
# print df_train

y_test = df_test.iloc[0:, 0:1].values
X_test = df_test.iloc[0:, 1:].values


classifier = tree.DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


df = pd.DataFrame(y_pred)
df.index += 1
df.index.name = 'PassengerID'
df.columns = ['Survived']
df.to_csv('titanicSurvival.csv', header=True)
print(y_pred)
