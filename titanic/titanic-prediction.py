import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv', index_col=[0])

# -------------------------------------------------------------
# separate independent and dependent variables
y = data['Survived']
X = data.loc[:,data.columns != 'Survived']

# -------------------------------------------------------------
# which columns should we keep
# Cabin, ticket, Name probably not useful
X = X.loc[:, ~X.columns.isin(['Name','Ticket', 'Cabin'])]


# -------------------------------------------------------------
# how do we handle missing values?
count_nan = X.isna().sum()

# for embarked use random sample from existing
X.loc[X['Embarked'].isna(), 'Embarked'] = X['Embarked'].sample(2).values

# for age use average age per sex
X.loc[(X['Sex']=='female') & (X['Age'].isna()), 'Age'] = X.loc[X['Sex']=='female', 'Age'].mean()
X.loc[(X['Sex']=='male') & (X['Age'].isna()), 'Age'] = X.loc[X['Sex']=='male', 'Age'].mean()

# -------------------------------------------------------------
# one hot encoding

temp = pd.get_dummies(X['Embarked'],prefix='Embarked', drop_first=True)
X = X.join(temp)
temp = pd.get_dummies(X['Sex'],prefix='Sex', drop_first=True)
X = X.join(temp)
temp = pd.get_dummies(X['Pclass'],prefix='Pclass', drop_first=True)
X = X.join(temp)

del X['Embarked']
del X['Sex']
del X['Pclass']

# -------------------------------------------------------------
# Separate test and train set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, 
                                                    shuffle=True, random_state=42)

# standardise the numeric values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train) # only fit on training set!!

X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# -------------------------------------------------------------
# Fit a regression model
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X,y)

# prediction
y_pred = clf.predict(X_test)
model_score = clf.score(X_test, y_test)


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
