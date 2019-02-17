import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn import linear_model

print("numpy",np.__version__)
print("matplotlib",matplotlib.__version__)
print("Pandas",pd.__version__)



#Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name)
    print (data)
    x_parameter =[]  #independent variable
    y_parameter =[]  #dependent variable
    for clothe_id,Age,Rating in zip(data['Clothing ID'],data['Age'],data['Rating']):
        if(clothe_id==1084):
            x_parameter.append([float(Age)])
            y_parameter.append([float(Rating)])
    return x_parameter,y_parameter
#x,y = get_data('house_price.csv')

#print x
#print y

def linear_model_main(X_parameters, Y_parameters,predict_value):
    regr = linear_model.LinearRegression()
    print (X_parameters)
    print (Y_parameters)
    regr.fit(X_parameters,Y_parameters)
    predict_outcome =  regr.predict(predict_value)
    pre = {}
    pre['intercept'] = regr.intercept_
    pre['coefficient'] = regr.coef_
    pre['predicted_value'] = predict_outcome
    plt.scatter(X_parameters,Y_parameters,color="m",marker='o',s=30)
    all_predicted_y = regr.predict(X_parameters)
    #plt.plot(X_parameters,Y_parameters,color="r")
    plt.plot(X_parameters, all_predicted_y, color="b")
    plt.scatter(predict_value,predict_outcome,color='g')
    print("INtercept value", pre['intercept'])
    print('coefficient', pre['coefficient'])
    print('pridicted value', pre['predicted_value'])
    plt.show()
    return pre
x , y =  get_data("Womens Clothing E-Commerce Reviews.csv")
predict_value=32
result = linear_model_main(x,y,predict_value)


#LogisticRegression
#LinearDiscriminantAnalysis
#GaussionNB
#DecisionClassifier
#KNeighbourClassifier
#svc
