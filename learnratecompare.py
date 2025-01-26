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

def gradient_descent(X, Y, theta, lr, iterations, threshold=1e-6):
    cost_history = []
    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - Y
        gradient = (X.T @ errors) / m
        theta -= lr * gradient
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)
        if i > 0 and abs(cost_history[-1] - cost_history[-2]) < threshold:
            print(f"Convergence achieved after {i + 1} iterations.")
            cost_history.extend([cost] * (iterations - len(cost_history)))
            break
    return theta, cost_history

learning_rate = 5 ######Mannualy changed learning rate for comparison######
iterations = 50
theta_optimal, cost_history = gradient_descent(X, Y, theta, learning_rate, iterations)

theta_denormalized = np.array([
    theta_optimal[0, 0] * Y_std + Y_mean - (theta_optimal[1, 0] * X_mean * Y_std / X_std),
    theta_optimal[1, 0] * Y_std / X_std
])

final_cost = cost_history[-1]

print("Final Parameters (Denormalized):")
print(f"theta_0 = {theta_denormalized[0]:.4f}")
print(f"theta_1 = {theta_denormalized[1]:.4f}")
print(f"Final Cost Function Value: {final_cost:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', color='b')
plt.title("Cost Function vs. Iterations", fontsize=14)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Cost Function", fontsize=12)
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(linearX, linearY, color='blue', label='Data Points')

x_line = np.linspace(min(linearX), max(linearX), 100)
y_line = theta_denormalized[0] + theta_denormalized[1] * x_line

plt.plot(x_line, y_line, color='red', label='Regression Line')
plt.title("Dataset and Fitted Regression Line", fontsize=14)
plt.xlabel("Independent Variable (X)", fontsize=12)
plt.ylabel("Dependent Variable (Y)", fontsize=12)
plt.legend()
plt.grid(True)
plt.show()
