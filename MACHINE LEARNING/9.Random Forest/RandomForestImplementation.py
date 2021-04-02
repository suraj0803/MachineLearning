import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


names = ['sepal-length','sepal-width','petal-length','petal-width','Class']
data = pd.read_csv("C:\\Users\\User\\Desktop\\9.Random Forest\\IRIS.csv",names=names)
print(data.head())

x = data.iloc[:, :-1].values
y = data.iloc[:, 4].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

classifier = RandomForestClassifier(n_estimators=50)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

res = confusion_matrix(y_test,y_pred)
print("Confusion Matrix : ",res)

res1 = classification_report(y_test,y_pred)
print("Classification Report : ",res1)

acc = accuracy_score(y_test,y_pred)
print("Accuracy : ",acc)




