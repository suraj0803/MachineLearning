#How Does K-Nearest Neighbors Work
#In short, K-Nearest Neighbors works by looking at the K closest points to the given data point (the one we want to classify) and 
#picking the class that occurs the most to be the predicted value. This is why this algorithm typically works best when we can identify
 #clusters of points in our data set (see below).

#Limitations and Drawbacks
#Although the KNN algorithm is very good at performing simple classification tasks it has many limitations.
 #One of which is its Training/Prediction Time. Since the algorithm finds the distance between the data point
 #and every point in the training set it is very computationally heavy. Unlike algorithms like linear regression which
  #simply apply a function to a given data point the KNN algorithm requires the entire data set to make a prediction. 
  #This means every time we make a prediction we must wait for the algorithm to compare our given data to each point. 
  #In data sets that contain millions of elements this is a HUGE drawback.

 
 #Another drawback of the algorithm is its memory usage. 
 #Due to the way it works (outlined above) it requires that the entire data set be loaded into memory to perform a prediction.
# It is possible to batch load our data into memory but that is extremely time consuming.

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import numpy as np 
import pandas as pd 
from sklearn import linear_model,preprocessing
#This will be used to normalize our data and convert non-numeric values into numeric values.

data = pd.read_csv("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\4 KNN\\car.data")
print(data.head())

le = preprocessing.LabelEncoder()
#The method fit_transform() takes a list (each of our columns) and will return to us an array containing our new values.
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clss = le.fit_transform(list(data["class"]))
#print(buying)

predict = "class"

#Now we need to recombine our data into a feature list and a label list. We can use the zip() function to makes things easier.
X = list(zip(buying,maint,door,persons,lug_boot,safety))#features
y = list(clss)#labels
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#print(x_train, y_test)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)

acc = model.score(x_test,y_test)
print(acc)

predicted = model.predict(x_test)
names = ["unacc","acc","good","vgood"]

for x in range(len(predicted)):
    print("Predicted :",names[predicted[x]], "Data : ",x_test[x], "Actual : " ,names[y_test[x]])
    n = model.kneighbors([x_test[x]],9,True)
    # The KNN model has a unique method that allows for us to see the neighbors of a given data point. We can use this information
    #  to plot our data and get a better idea of where our model may lack accuracy. We can use model.neighbors to do this.
    # Note: the .neighbors method takes 2D as input, this means if we want to pass one data point we need surround it with []
    #  so that it is in the right shape.
    #Parameters: The parameters for .neighbors are as follows: data(2D array), # of neighbors(int), distance(True or False)
    #Return: This will return to us an array with the index in our data of each neighbor. If distance=True then 
    # it will also return the distance to each neighbor from our data point.
    print(n)


#Summary
#The K-Nearest Neighbor algorithm is very good at classification on small data sets that contain few dimensions (features). 
#It is very simple to implement and is a good choice for performing quick classification on small data. 
#However, when moving into extremely large data sets and making a large amount of predictions it is very limited.