# Saving and Training Multiple Models.

# Import Modules
import numpy as np  
import pandas as pd  
from sklearn import linear_model
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use("ggplot")

data = pd.read_csv("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\student-mat.csv",sep = ";")

predict = "G3"

data = data[["G1","G2","absences","failures","studytime","G3"]]
data = shuffle(data) # Optional-shuffle the data

x = np.array(data.drop([predict],1))
y = np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
linear = linear_model.LinearRegression()

linear.fit(x_train,y_train)
acc = linear.score(x_test,y_test)
print(acc)


# Saving our Model
# To save out model we will write to a new file using pickle.dump()

with open("studentmodel.pickle","wb") as f:
    pickle.dump(linear,f)

# Loading our Model
#Once we have save our model we can load it in using the following two lines. Now you can remove the code that creates and trains our model
# as we are siply loading in an exitsting one from our pickle file.

pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)
# Now we can use linear to predict grades like before

#Training multiple models
# You may have noticed that our models vary in accuracy. This is because when we split the data into training and testing  it is divided
# differently each time. Since our model train very quickly it may be worth training multiple models and saving the best one.

#Train model multiple times for Best Score.
best = 0
for _ in range(20):
    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print("Accuracy :"+str(acc))
    # "If the current model has the better score than we'have already trained then save it"
    if acc > best:
        best = acc
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)

# Plotting our Data
# To get a visual representation of data we van plot it using the matplotlib library we installed earlier We are going to use a scatter plot to visualize our data.
plot = "failures" #change this to G1,G2 ,studypoint or absences to see other graphs
plt.scatter(data[plot],data["G3"])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel("Final Grade")
plt.show()
