import pandas as pd 
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

names = ['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']

data = pd.read_csv("C:\\Users\\User\\Desktop\\DecisionTree\\pima-indians-diabetes.csv",header=None,names=names)
print(data.head())

# Now split the dataset into feature and tRGETS
feature = ['pregnant','insulin','bmi','age','glucose','bp','pedigree']

x = data[feature] #Features
y = data.label  #Target Variable

# Now we divide the data into train and test split. 
#We will split 90% training data and 10% test data.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)

# Next train the model with the help of DecisionTreeClassifier class of sklearn 
clf = DecisionTreeClassifier()
clf = clf.fit(x_train,y_train)

# Atlast we make the prediction
y_pred = clf.predict(x_test)

# Now we get the accuracy score, classification report and confusion matrix.
res = confusion_matrix(y_test, y_pred)
print("Confusion Matrix : ",res)

res1 = classification_report(y_test, y_pred)
print("Classification Report : ",res1)

acc = accuracy_score(y_test,y_pred)
print("Accuracy: ",acc)

