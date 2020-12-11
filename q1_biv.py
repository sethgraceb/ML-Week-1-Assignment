import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)
from sklearn.linear_model import LinearRegression

dataf = pd.read_csv("assignment1.csv", comment = '#')
X = dataf.iloc[:, 0].values.reshape(-1, 1)
Y = dataf.iloc[:, 1].values.reshape(-1, 1)
#print(dataf)
linear_regression = LinearRegression()
linear_regression.fit(X, Y)			#linear regression
prediction = linear_regression.predict(X)		#make predictions

plt.scatter(X, Y)
plt.gca().set_title("Gradient Descent Linear Regression", color = 'black')
plt.xlabel("X", color = 'blue'); plt.ylabel("Y", color = 'blue')
plt.plot(X, prediction, color = 'red')
plt.legend(["prediction", "training data"])
plt.show()