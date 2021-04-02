#Univariate Selection : This feautre selection techniques is very useful in selecting those features with the help of statistical testing,
#having strongest relationship with the prediction variables.

#We can Implement univariate selection technique with the help of SelectKBestt0class of scikit-learn Python Library.


# Ex: In this example, we will use Pima Indian Diabetes dataset to select 4 of the attributes having best feature with the help of chi_square statistical test.

from pandas import read_csv
from numpy import set_printoptions

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

path = ("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\MLConcepts\\pima-indians-diabetes.csv")

names = ['preg' ,'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age','class' ]

dataframe = read_csv(path,names=names)
array = dataframe.values

#Next, we will seperate arrays into input and output components:
X = array[:,0:8]
Y = array[:,8]

#The following line of code will select the best feature from dataset.

test = SelectKBest(score_func=chi2,k=4)
fit = test.fit(X,Y)

#We can also summarize the data for outputs as per our choice. Here we are setting the precision to 2 and showing the 4 data attributes with the
#best feature along with best score of each attribute.

set_printoptions(precision=2)
print(fit.scores_)
featured_data = fit.transform(X)
print("\nFeatured_data:\n",featured_data)