# correct imbalance

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv', index_col=[0])

# make sure we have equal amounts of survived and not survived
# upsample survivals
from sklearn.utils import resample

df_majority = data.loc[data['Survived']==0,'Survived']
df_minority = data.loc[data['Survived']==1,'Survived']
req = (len(df_majority) - len(df_minority))

df_minority_resampled = resample(df_minority, replace=True,
                                 n_samples=req, random_state=42)

new_data = pd.concat([data, data.loc[df_minority_resampled.index]])

# -------------------------------------------------------------
# seperate independent and dependent variables
y = new_data['Survived']
X = new_data.loc[:,new_data.columns != 'Survived']

# -------------------------------------------------------------
# which columns should we keep
# Cabin, ticket, Name probably not useful
X = X.loc[:, ~X.columns.isin(['Name','Ticket', 'Cabin'])]


# -------------------------------------------------------------
# how do we handle missing values?
count_nan = X.isna().sum()

# for embarked use random sample from existing
X.loc[X['Embarked'].isna(), 'Embarked'] = X['Embarked'].sample(count_nan['Embarked']).values

# for age use average age per sex
X.loc[(X['Sex']=='female') & (X['Age'].isna()), 'Age'] = X.loc[X['Sex']=='female', 'Age'].mean()
X.loc[(X['Sex']=='male') & (X['Age'].isna()), 'Age'] = X.loc[X['Sex']=='male', 'Age'].mean()

# -------------------------------------------------------------
# one hot encoding
# categorical variables need to be encoded
cat_var = ['Pclass','Sex', 'Embarked']

# make Sex numeric male=0, female=1
X.loc[X['Sex']=='female', 'Sex'] = 1
X.loc[X['Sex']=='male', 'Sex'] = 0

# embarked, C=0, Q=1, S=2
X.loc[X['Embarked']=='C', 'Embarked'] = 0
X.loc[X['Embarked']=='Q', 'Embarked'] = 1
X.loc[X['Embarked']=='S', 'Embarked'] = 2

# onehotencode
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(sparse=False)
X_onehot = pd.DataFrame(enc.fit_transform(X[cat_var]))

X_onehot.columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_M', 'Sex_F', 
                    'Embarked_C', 'Embarked_Q', 'Embarked_S']
X_onehot.index = X.index

# we can drop one of each -- it will be implied by the zeros in the other columns
del X_onehot['Sex_F']
del X_onehot['Pclass_3']
del X_onehot['Embarked_S']

# recombine with main X
del X['Pclass']
del X['Sex']
del X['Embarked']

X[X_onehot.columns] = X_onehot

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



# store coefficient weights
coeff = pd.DataFrame(clf.coef_.T, index = X.columns)

# -------------------------------------------------------------
# new model
var= ['Pclass_1',
    'Pclass_2',
    'Sex_M',
    'Embarked_C',
    'Embarked_Q',]

clf.fit(X[var], y)
clf.score(X_test[var], y_test)

pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)

# store coefficient weights
coeff = pd.DataFrame(clf.coef_.T, index = X[var].columns)
