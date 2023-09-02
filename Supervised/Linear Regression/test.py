import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression
from sklearn.linear_model import LinearRegression as LR

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

reg = LinearRegression(lr=0.01)
reg.lr = reg.learning_rate_recommender(X_train, X_test, y_train, y_test, rate=0.005)
reg.fit(X_train,y_train)
predictions = reg.predict(X_test)
print(reg.weight)
print(reg.bias)

reg_sci = LR()
reg_sci.fit(X_train, y_train)
y_pred_line2 = reg_sci.predict(X_test)
print(reg_sci.coef_)
print(reg_sci.intercept_)

y_pred_line = reg.predict(X_test)

plt.subplot(1,2,1)
plt.scatter(X_test, y_test, s=10)
plt.plot(X_test, y_pred_line, 'b-.')
plt.grid('on')

plt.subplot(1,2,2)
plt.scatter(X_test, y_test, s=10)
plt.plot(X_test, y_pred_line2, 'r-.')
plt.grid('on')
plt.show()