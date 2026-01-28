import numpy as np

def sigmoid(z):
	return (1 / (1+np.exp(-z)))

def MSE(pred, y):
	return np.mean((pred -y) ** 2)
	
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	X = np.array(features)
	y = np.array(labels)
	updated_weights = np.array(initial_weights)
	updated_bias = initial_bias
	mse_values = []
	for _ in range(epochs):
		z = np.dot(X, updated_weights) + updated_bias

		pred = sigmoid(z)

		mse_values.append(round(MSE(pred, y),4))

		dz = 2 * (pred - y) * pred * (1 - pred)

		dw = np.mean(dz[:, np.newaxis] * X, axis = 0)
		db = np.mean(dz)

		updated_weights -= learning_rate * dw
		updated_bias -= learning_rate * db


	return np.round(updated_weights, 4), np.round(updated_bias, 4), mse_values
