import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import  MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("titanic.csv")

df.head(4)
# for random daTa
df.sample(5)
df.info()
# find out the missing value in the data set
df.isnull().sum()
# how doese the data look mathematically
df.describe()
df.drop(columns=["PassengerId","Name","Ticket","Cabin"],inplace=True)
df.head(4)

x_train,x_test,y_train,y_test = train_test_split(df.drop(columns=["Survived"]),df["Survived"],
                                                 test_size=0.2,random_state=42)

x_train.head(2)

df.isnull().sum()

# Here we find multiple nan values so we doo imputation to wipe out the nan value
# applying imputation

si_age = SimpleImputer()
si_embarked = SimpleImputer(strategy="most_frequent")

x_train_age = si_age.fit_transform(x_train[["Age"]])
x_train_embarked = si_embarked.fit_transform(x_train[["Embarked"]])

x_test_age = si_age.transform(x_test[["Age"]])
x_test_embarked = si_embarked.transform(x_test[["Embarked"]])

# OneHotEncoding

ohe_sex = OneHotEncoder(sparse_output=False,handle_unknown="ignore")
ohe_embarked = OneHotEncoder(sparse_output=False,handle_unknown="ignore")

x_train_sex = ohe_sex.fit_transform(x_train[["Sex"]])
x_train_embarked = ohe_embarked.fit_transform(x_train_embarked)

x_test_sex = ohe_sex.transform(x_test[["Sex"]])
x_test_embarked = ohe_embarked.transform(x_test_embarked)

x_train_rem = x_train.drop(columns=["Sex","Age","Embarked"])
x_test_rem = x_test.drop(columns=["Sex","Age","Embarked"])

x_train_transformed = np.concatenate((x_train_rem,x_train_age,x_train_sex,x_train_embarked),axis=1)
x_test_transformed = np.concatenate((x_test_rem,x_test_age,x_test_sex,x_test_embarked),axis=1)
x_train_transformed

# Decison Tree
clf =  DecisionTreeClassifier()
clf.fit(x_train_transformed,y_train)

y_pred = clf.predict(x_test_transformed)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
import pickle
pickle.dump(ohe_sex,open("ohe_sex.pkl","wb"))
pickle.dump(ohe_embarked,open("ohe_embarked.pkl","wb"))
pickle.dump(clf,open("clf.pkl","wb"))

ohe_sex = pickle.load(open('ohe_sex.pkl','rb'))
ohe_embarked = pickle.load(open('ohe_embarked.pkl','rb'))
clf = pickle.load(open('clf.pkl','rb'))

test_input = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'],dtype=object).reshape(1,7)

test_input_sex = ohe_sex.transform(test_input[:,1].reshape(1,1))

test_input_embarked = ohe_embarked.transform(test_input[:,-1].reshape(1,1))

test_input_age = test_input[:,2].reshape(1,1)
test_input_transformed = np.concatenate((test_input[:,[0,3,4,5]],test_input_age,test_input_sex,test_input_embarked),axis=1)
test_input_transformed.shape
clf.predict(test_input_transformed)
