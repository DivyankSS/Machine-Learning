import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:3].values

from sklearn.tree import DecisionTreeRegressor
DT_reg=DecisionTreeRegressor(random_state=0)
DT_reg.fit(x,y)



# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, DT_reg.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVM_SVC_Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

y_pred = DT_reg.predict(np.array([[6.5]]))
print(y_pred)

x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, DT_reg.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()