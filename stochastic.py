import numpy as np
import matplotlib.pyplot as plt

linearX = np.loadtxt('linearX.csv', delimiter=',')
linearY = np.loadtxt('linearY.csv', delimiter=',')

X_mean, X_std = np.mean(linearX), np.std(linearX)
Y_mean, Y_std = np.mean(linearY), np.std(linearY)

X_normalized = (linearX - X_mean) / X_std
Y_normalized = (linearY - Y_mean) / Y_std

X = np.column_stack((np.ones(X_normalized.shape[0]), X_normalized))
Y = Y_normalized.reshape(-1, 1)

theta = np.zeros((X.shape[1], 1))
m = len(Y)

def stochastic_gradient_descent(X, Y, theta, lr, iterations):
    cost_history = []
    for i in range(iterations):
        for j in range(m):
            rand_index = np.random.randint(m)
            x_i = X[rand_index, :].reshape(1, -1)
            y_i = Y[rand_index].reshape(1, 1)

            prediction = x_i @ theta
            error = prediction - y_i
            gradient = x_i.T @ error

            theta -= lr * gradient

        cost = np.sum((X @ theta - Y) ** 2) / (2 * m)
        cost_history.append(cost)

    return theta, cost_history

learning_rate = 0.5
iterations = 50
theta_optimal, cost_history = stochastic_gradient_descent(X, Y, theta, learning_rate, iterations)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', color='r')
plt.title("Cost Function vs. Iterations (SGD)", fontsize=14)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Cost Function", fontsize=12)
plt.grid(True)
plt.show()
