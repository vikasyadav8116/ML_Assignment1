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
batch_size = 32

def mini_batch_gradient_descent(X, Y, theta, lr, iterations, batch_size):
    cost_history = []
    for i in range(iterations):
        permutation = np.random.permutation(m)
        X_shuffled = X[permutation]
        Y_shuffled = Y[permutation]

        for j in range(0, m, batch_size):
            X_batch = X_shuffled[j:j + batch_size]
            Y_batch = Y_shuffled[j:j + batch_size]
            predictions = X_batch @ theta
            errors = predictions - Y_batch
            gradient = (X_batch.T @ errors) / batch_size
            theta -= lr * gradient

        cost = np.sum(errors ** 2) / (2 * m)
        cost_history.append(cost)

    return theta, cost_history

learning_rate = 0.5
iterations = 50
theta_optimal, cost_history = mini_batch_gradient_descent(X, Y, theta, learning_rate, iterations, batch_size)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', color='r')
plt.title("Cost Function vs. Iterations (Mini-Batch GD)", fontsize=14)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Cost Function", fontsize=12)
plt.grid(True)
plt.show()
