#Skewness is defined as the distribution that is assumed to be guassianbut appears distorted or shifted in one direction or another,or wither to the left or right.

# Presence of skewness in data requires the correction at data preparation stage so that we can get more accuracy from our model.

from pandas import read_csv
path = ("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\MLConcepts\\pima-indians-diabetes.csv")

names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

data = read_csv(path, names=names)
print(data.skew())