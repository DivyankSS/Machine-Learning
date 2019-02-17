
import pandas
fillename='indians-diabetes.data.csv'
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data= pandas.read_csv(fillename,names=name)
data.head(20)
print(data)
print (data.dtypes)
pandas.set_option('display.width',150)
pandas.reset_option('precision',3)
print (data.describe())

class_counts=data.groupby('class').size()
print (class_counts)
correaltion=data.corr(method='pearson')
print (correaltion)
#=============================================================
'''

from matplotlib import pyplot
import pandas
fillename='indians-diabetes.data.csv'
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data= pandas.read_csv(fillename,names=name)
data.hist()
pyplot.show()

#univariant plot
#multivariate plot


#Univariate histogram plot
#univariant Density Plot
#univariate Box and Whisker Plot

from matplotlib import pyplot
import pandas
fillename='indians-diabetes.data.csv'
name=['preg','plas','pres','skin','test','mass','pedi','age','class']
data= pandas.read_csv(fillename,names=name)
data.plot(kind='density',subplots=True,layout=(3,3),sharex=False)
pyplot.show()

'''