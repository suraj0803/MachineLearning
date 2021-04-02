# Correlation Between Attributes.(The relationship between two values is called correlation.)
# Coefficient value = 1 : it represents full positive correlation between varibles.
# Coefficient value = -1 : it represents full negative correlation between variables. 
# Coefficient value = 0 : it represents no correlation at all between variables.

from pandas import read_csv
from pandas import set_option

path = ("C:\\Users\\User\\Desktop\\MACHINE LEARNING\\MLConcepts\\pima-indians-diabetes.csv")

names = ['preg','plas','pres','skin','test','mass','pedi','pedi','age','class']
data = read_csv(path , names=names)

set_option('display.width',100)
set_option('precision',2)

correlations = data.corr(method='pearson')

print(correlations)