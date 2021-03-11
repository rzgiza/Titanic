import numpy as np
import pandas as pd 

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

train_data = pd.read_csv("data/train.csv")
test_data = pd.read_csv("data/test.csv")

################### Training Data Quick EA ####################################
#Dataframe Infromation
size = train_data.size 
shape = train_data.shape 
df_ndim = train_data.ndim 

#printing size and shape
print("\n.............Training Data EA ...............\nDataframe size and shape:")
print("-------------")
print("Size = {}\nShape ={}\nShape[0] x Shape[1] = {}".format(
        size, shape, shape[0]*shape[1]
        )) 

#printing ndim 
print("ndim of dataframe = {}\n".format(df_ndim))

#Columns with (missing count)
print("Data Columns (Nan Count):")
print("-------------")
for col in train_data.columns.to_list():
    print(col, "(Missing: " + str(train_data[col].isna().sum()) + ")")
    
#Data type info
print("\nData Info:")
print("-------------")
print(train_data.info())

######################### Feature Engineering #################################
X = train_data.drop(["PassengerId", "Survived", 'Name', 'Ticket', 'Cabin'], axis=1)
y = train_data["Survived"]
X["Pclass"] = X["Pclass"].astype(str)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
print("\n.............Varaible Selection...............")
print("\nSelected Variables:")
print("-------------")
print("Numerical:\n", numeric_features, "\n")
print("Categorical:\n", categorical_features, "\n")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
    ])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
    ]) 

print("\nData Imputer and Variable Transformers using Pipeline:")
print("-------------")
print("\n[Numerical Variables]:\n")
print(numeric_transformer)
print("\n[Categorical Variables]:\n")
print(categorical_transformer)
########################## Model Selection ####################################
print("\n.............Model Selection using Pipeline...............")
classifiers = [
    GaussianNB(),
    LogisticRegression(solver='lbfgs'),
    KNeighborsClassifier(n_neighbors=10) 
    ]
for classifier in classifiers:
    pipe = Pipeline(steps=[
               ('preprocessor', preprocessor),
               ('classifier', classifier)
               ])
    scores = cross_val_score(pipe, X, y, cv=5)
    print("\n", classifier)
    print('''
          CV mean score {:0.2f}%
          CV std {:0.2f}% 
          CV max score {:05.2f}% 
          CV min score {:05.2f}%'''.format(
              scores.mean()*100,
              scores.std(ddof=1)*100,
              scores.max()*100,
              scores.min()*100
              ))
    print("--------------------------------------------")

################## Fitting Model on Test Data and Predictions #################
#Fitting All Training Data and Predicting on Test Data
Xtest = test_data.drop(["PassengerId", 'Name', 'Ticket', 'Cabin'], axis=1)
Id = test_data["PassengerId"]
pipe = Pipeline(steps=[
                   ('preprocessor', preprocessor),
                   ('KNN_10', KNeighborsClassifier(n_neighbors=10))
                   ])
pipe.fit(X, y)
ypred = pipe.predict(Xtest)

#Formatting predictions and creating output file
id_pred = pd.DataFrame(
                np.concatenate((Id.values.reshape(-1, 1), 
                ypred.reshape(-1, 1)), axis=1), 
                columns=["PassengerId", "Survived"]
                )
id_pred.to_csv('output/predictions.csv', index=False)

