#Checking Dimension and datatypes
# use data.dtypes to check datatypes
# data.shape->to check dimension
# describe() is a function of pandas
             # Count,mean,Standard Deviation,Minimum value,Maximum value,25%,Median i.e.50%,75%.

from pandas import read_csv
path = ("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\MLConcepts\\IRIS.csv")
data = read_csv(path)
print(data.shape)
print(data.dtypes)
print(data.describe())


# output: 99 rows and 9 columns
