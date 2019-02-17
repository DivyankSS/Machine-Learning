import numpy
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values

print(x)
print(y)

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
L_reg=LinearRegression()
L_reg=L_reg.fit(x,y)
y_pre=L_reg.predict(x)

plt.scatter(x,y)
plt.plot(x,y)
plt.scatter(x,y_pre,color='red')
plt.plot(x,y_pre)
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)
#poly_reg=poly_reg.fit(x_poly,y)
L_reg2=LinearRegression()
L_reg2=L_reg2.fit(x_poly,y)
y_poly_pre=L_reg2.predict(x_poly)

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, y_poly_pre, color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#print(L_reg2.predict([[6.5]]))