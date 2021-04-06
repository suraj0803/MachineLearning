# Random Forest

import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv("C:\\Users\\User\\Desktop\\Bagging\\pima-indians-diabetes.csv", names=names)

print(data.head())

array = data.values 
X = array[:,0:8]
Y = array[:,8]

# Give the input for 10-fold cross validation
seed = 7
kfold = KFold(n_splits=10,random_state=7, shuffle=True)

# We need to provide the number of trees we are going to build. Here we are building 150 trees.
num_trees = 150
max_features = 5

# Build the model with the help of script
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

#Calculate and print the results.
results = cross_val_score(model,X,Y,cv=kfold)
print(results.mean())