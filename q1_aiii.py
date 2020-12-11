import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 5.0)

dataf = pd.read_csv("assignment1.csv", comment = '#')
X = dataf.iloc[:, 0]
Y = dataf.iloc[:, 1]

theta = np.zeros(2)
m = len(Y)
n = len(X)
x = np.column_stack((np.ones(n), X))  #stack into columns

#--------Gradient Descent--------------
iterations = 1	
alpha = 0.001		#0.01 0.1

for i in range(iterations):
	t0 = theta[0] - ((2 * alpha) / m) * np.sum(np.dot(x, theta) - Y)
	t1 = theta[1] - ((2 * alpha) / m) * np.sum((np.dot(x, theta) - Y) * x[:, 0])
	theta = np.array([t0, t1])
print('theta:', theta)

cost = np.sum(np.square(np.matmul(x, theta) - Y)) / (1 * m)	#cost function
print('cost:', cost)

plt.scatter(X, Y)	
plt.gca().set_title("Gradient Descent Linear Regression", color = 'black')
plt.xlabel("X", color = 'blue'); plt.ylabel("Y", color = 'blue')
prediction = np.linspace(min(X), max(X))
plt.plot(prediction, theta[0] + theta[1] * prediction, color = 'red')
plt.legend(["prediction","training data"])
plt.show()
