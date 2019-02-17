import csv
import numpy
import urllib.request
import pandas

'''
filename=open("indians-diabetes.data.csv",'r')
reader=csv.reader(filename,delimiter=',')
x= list(reader)
print(x)

filename="indians-diabetes.data.csv"
raw_data=open(filename,'rb')
data=numpy.loadtxt(raw_data,delimiter=',')
print("numpy loadtext:",data.shape)
print(data)

web_path=urllib.request.urlopen("https://goo.gl/QnHW4g")
dataset=numpy.genfromtxt(web_path,delimiter=',')
print("Shap",dataset.shape)
print(dataset)

'''
filename="indians-diabetes.data.csv"
names=['preg','plas','pres','skin','test','mass','pedi','age','class']
data=pandas.read_csv(filename,names=names)
print(data.shape)
print(data.head())

'''

url="https://goo.gl/QnHW4g"
name=['a','b','c','d']
data=pandas.read_csv(url,names=name)
print(data.shape)
print(data)
'''