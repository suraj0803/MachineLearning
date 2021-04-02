from pandas import read_csv

path = ("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\MLConcepts\\IRIS.csv")
data = read_csv(path)
print(data.shape)
print(data[:3])