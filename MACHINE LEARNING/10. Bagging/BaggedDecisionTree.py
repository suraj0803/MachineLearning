# The bagging ensemble works well with the algorithms that have high variance and, the best one is
# Decision Tree Algorithm.

import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv("C:\\Users\\User\\Desktop\\Bagging\\pima-indians-diabetes.csv", names=names)

print(data.head())

array = data.values 
X = array[:,0:8]
Y = array[:,8]

# Give the input for 10-fold cross validation
seed = 7
kfold = KFold(n_splits=10,random_state=7, shuffle=True)
cart = DecisionTreeClassifier()

# We need to provide the number of trees we are going to build. Here we are building 150 trees.
num_trees = 150

# Build the model with the help of script
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)

#Calculate and print the results.
results = cross_val_score(model,X,Y,cv=kfold)
print(results.mean())