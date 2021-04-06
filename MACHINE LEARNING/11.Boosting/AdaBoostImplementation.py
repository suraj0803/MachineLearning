# AdaBoost
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

names = ['preg','plas','pres','skin','test','mass','pedi','age','class']
data = pd.read_csv("C:\\Users\\User\\Desktop\\Boosting\\pima-indians-diabetes.csv", names=names)

print(data.head())

array = data.values 
X = array[:,0:8]
Y = array[:,8]

# Give the input for 10-fold cross validation
seed = 5
kfold = KFold(n_splits=10,random_state=seed,shuffle=True)

# We need to provide the number of trees we are going to build. Here we are building 150 trees.
num_trees = 50

# Build the model with the help of script
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)

#Calculate and print the results.
results = cross_val_score(model,X,Y,cv=kfold)
print(results.mean())