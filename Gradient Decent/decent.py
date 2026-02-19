import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Generate Synthetic Data
np.random.seed(42)
n_samples = 100
X = np.linspace(0, 10, n_samples)
y = 2 * X + 2 + np.random.randn(n_samples)

# Visualize the data

plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.show()

#Reshape X and y arrays

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd = SGDRegressor(alpha=0.01, max_iter=1000, tol=1e-3)
sgd.fit(X_train, y_train.ravel())

y_pred = sgd.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Test set MSE: {mse}')
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color='red')
plt.show().xlabel('X')
plt.ylabel('y')
plt.show()