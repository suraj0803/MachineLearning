from numpy import loadtxt
path=("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\MLConcepts\\pima-indians-diabetes.csv")

datapath = open(path,'r')
data = loadtxt(datapath,delimiter=",")
print(data.shape)
print(data[:3])