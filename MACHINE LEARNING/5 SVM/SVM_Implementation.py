#SVM stands for a support vector machine. SVM's are typically used for classification tasks similar to what we did with K Nearest Neighbors. 
# They work very well for high dimensional data and are allow for us to classify data that does not have a linear correspondence
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)
print(x_train, y_train)

classes = ['malignant', 'benign']
#Adding a Kernel
#The reason we received such a low accuracy score was we forgot to add a kernel! We need to specify which kernel we should use to increase our accuracy.
#Kernel Options:
#- linear
#- poly
#- rbf
#- sigmoid
#- precomputed
#We will use linear for this data-set.

#Changing the Margin
#By default our kernel has a soft margin of value 1. This parameter is known as C.
 #We can increase C to give more of a soft margin, we can also decrease it to 0 to make a hard margin.
  #Playing with this value should alter your results slightly.
clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train,y_train)

y_pred  =clf.predict(x_test)

acc = metrics.accuracy_score(y_test,y_pred)
print(acc)