# Import Modules/Packages
import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

# Loading in our data
data = pd.read_csv("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\student-mat.csv", sep=";")
#since our data is seperated by semicolon we need to do sep=";"

#To see our frame we can type
#print(data.head())
#print(data.shape)
#print(data.describe())
#print(data.dtypes)

# Trimming our data : Since we have so many attributes and not all are relevant we need to select the one we want to use.
data = data[["G1","G2","G3","studytime","failures","absences"]]
#Now our data frame only has the information associated with those 6 attributes.

#Seperating the data: Now that we have trimmed our data set down we need to seperate it into 4 arrays.However before we can do that what attribute we are trying to predict.
#This attribute is known as label. The other attribute that will define our label is know as feature. Once we have done this we 
# will use numpy to create two arrays. One that contains all of out feature and other that contains all our labels.

predict = "G3"
X = np.array(data.drop([predict],1)) #features
Y = np.array(data[predict])#labels

#After this we will need to split our data into testing and training data. We will use 90% of our data to train and the other 10% to test.
#The reason we do this is so that we donot testour model our data that it has already seen.

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X,Y,test_size=0.1)

# Implementation of Lnear Regression
# Linear regression is essentially just a best fit line . Given a set of data the algorithm will create a best fit line through these data points.
#This line can be defined by the equation y = m*x + b.

#m is the slope. Meaning how much the y value increases for each x value.
#b is the y intercept. Where the line crosses the y axis.
#We can determine the slope(m) of the line by picking two points on the line (p1 and p2) and using the following equation: m = (y2 - y1) / (x2 - x1)
# once the computer has generated this line it will use it to predict certain values.
#In reality most of our best fit lines will span across multiple dimensions and therefore will have multiple slope values.

# We will start by defining the model which we will be using.
linear = linear_model.LinearRegression()
# Next we will train and score our model using the arrays we created.
linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)# acc stands for accuracy.
# To see how well our algorithm performed on our test data we can print out the accuracy
print(acc)

# Viewing the constraints : I we want to see the constraints used to generate the line we can type the following:
print("Coefficient\n",linear.coef_)# these are each slope value
print("Intercept \n",linear.intercept_)#this is the intercept

# Predicting on specific students :Seeing a score value is cool but I'd like to see how well our algorithm works on specific students.
#  To do this we are going to print out all of our test data.
#  Beside this data we will print the actual final grade and our models predicted grade.

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x]) # Get a list of all prediction

