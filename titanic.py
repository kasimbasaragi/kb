## titanic dataset solution 
#importing all lib which req.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#read the csv file which we download before
train = pd.read_csv('Downloads/titanic/train.csv')

#to konw shape of df
train.shape #...(891, 12)

#to display df in ongoing code..
train.head()

#to check null value in column
train.isnull()

#to show all column elements
train.columns.values

#it shows all summery info regarding df
train.info()

#it shows heat map of above df
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#it shows count plot of df
sns.set_style('whitegrid')
sns.countplot(x= 'Survived',data=train)

#it shows sex vise distribution in count plot
sns.set_style('whitegrid')
sns.countplot(x= 'Survived',hue='Sex',data=train)

#it shows survived vise distribution in count plot
sns.set_style('whitegrid')
sns.countplot(x= 'Survived',hue='Pclass',data=train)

#dist plot age vise
sns.distplot(train['Age'].dropna(),kde=False,color='b',bins=70)

#it shows count plot of SibSp
sns.countplot(x='SibSp', data=train)

#plot for fare
train['Fare'].hist(color='green',bins=40,figsize=(8,5))

#box plot for age vs passenger class
plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass', y='Age', data=train,palette='winter')

#to fill null values in df we fill accordinglly passenger class we take mean age value of each class
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

#imputing null age as mean age value      
train['Age'] = train[['Age', 'Pclass']].apply(impute_age,axis=1) 

#head map for checkning age and cabine checking
sns.heatmap(train.isnull(),yticklabels=False,cmap='viridis')

train.head()

#drop the column which have null value in column
train.dropna(inplace=True)

train.info()

#creating dummies for null value in embarked column and drop the first column 
pd.get_dummies(train['Embarked'],drop_first=True).head()

##creating dummies for null value in sex column and drop the first column 
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

#drop sex, embarked, name, ticket column
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train.head()

#concading column sex and embark
train = pd.concat([train,sex,embark],axis=1)

train.head()

#droppning survived column
train.drop('Survived',axis=1).head()

train['Survived'].head()

#model selection
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),
                                                   train['Survived'], test_size=0.30,
                                                   random_state=101)
#linear regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#predictions
predictions = logmodel.predict(X_test)
from sklearn.metrics import confusion_matrix
accuracy=confusion_matrix(y_test,predictions)

accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,predictions) #####.....score- 0.797752808988764
accuracy

predictions

#output- array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1,
       1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,
       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
       0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0,
       1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1,
       0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
       0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0,
       0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0,
       1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0,
       0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
       0, 1, 1], dtype=int64)
