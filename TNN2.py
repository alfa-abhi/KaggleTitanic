import warnings

import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

warnings.filterwarnings("ignore", category=DeprecationWarning)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df = df_train.append(df_test, ignore_index=True)
df['Title'] = df.Name.map(lambda x: x.split(',')[1].split('.')[0].strip())
df['Title'] = df['Title'].replace('Mlle', 'Miss')
df['Title'] = df['Title'].replace(['Mme', 'Lady', 'Ms'], 'Mrs')
df.Title.loc[(df.Title != 'Master') & (df.Title != 'Mr') & (df.Title != 'Miss') & (df.Title != 'Mrs')] = 'Others'
df = pd.concat([df, pd.get_dummies(df['Title'])], axis=1).drop(labels=['Name'], axis=1)

df.Sex = df.Sex.map({'male': 0, 'female': 1})
df['Family'] = df['SibSp'] + df['Parch'] + 1
df.Family = df.Family.map(lambda x: 0 if x > 4 else x)
df.Ticket = df.Ticket.map(lambda x: x[0])
guess_Fare = df.Fare.loc[(df.Ticket == '3') & (df.Pclass == 3) & (df.Embarked == 'S')].median()
df.Fare.fillna(guess_Fare, inplace=True)

df['Fare-bin'] = pd.qcut(df.Fare, 5, labels=[1, 2, 3, 4, 5]).astype(int)


# exit(0)

df.Embarked.fillna('C', inplace=True)
df.Embarked = df.Embarked.map({'S': 0, 'C': 1, 'Q': 2})


print df
# df = df.drop(labels='Embarked', axis=1)
df_sub = df[['Age', 'Master', 'Miss', 'Mr', 'Mrs', 'Others', 'SibSp', 'Fare-bin']]

df.loc[(df.Title == 'Master') & (df['Age'].isnull()), ['Age']] = 4.57
df.loc[(df.Title == 'Miss') & (df['Age'].isnull()), ['Age']] = 21.85
df.loc[(df.Title == 'Mr') & (df['Age'].isnull()), 'Age'] = 32.37
df.loc[(df.Title == 'Mrs') & (df['Age'].isnull()), 'Age'] = 35.90
df.loc[(df.Title == 'Others') & (df['Age'].isnull()), 'Age'] = 45.43

bins = [0, 4, 12, 18, 30, 50, 65, 100]  # This is not somewhat arbitrary...
age_index = (1, 2, 3, 4, 5, 6, 7)
# ('baby','child','teenager','young','mid-age','over-50','senior')
df['Age-bin'] = pd.cut(df.Age, bins, labels=age_index).astype(int)

df['Ticket'] = df['Ticket'].replace(['A', 'W', 'F', '7', 'L', '6', '8', '9'], '4')
# df['Ticket'] = df['Ticket'].replace([], '5')

df = pd.get_dummies(df, columns=['Ticket'])
df = pd.get_dummies(df, columns=['Embarked'])
# df = pd.get_dummies(df, columns=['Sex'])
df = pd.get_dummies(df, columns=['Pclass'])

df = df.drop(labels=['SibSp', 'Parch', 'Title', 'Cabin', 'Age', 'Fare'], axis=1)

print df

y_train = df[0:891]['Survived'].values
X_train = df[0:891].drop(['Survived', 'PassengerId'], axis=1).values
X_test = df[891:].drop(['Survived', 'PassengerId'], axis=1).values

model = Sequential()

model.add(Dense(units=9, kernel_initializer='uniform', activation='relu', input_dim=23))
model.add(Dense(units=9, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=400)

y_pred = model.predict(X_test)
y_final = (y_pred > 0.5).astype(int).reshape(X_test.shape[0])

output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': y_final})
output.to_csv('predictionLSD.csv', index=False)
