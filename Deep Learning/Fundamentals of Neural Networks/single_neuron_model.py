import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	probabilities = []
	for x in features:
		z = sum(f * w for f, w in zip(x, weights)) + bias
		prob = 1 / (1 + math.exp(-z))
		probabilities.append(round(prob, 4))
	total_squared_error = sum((p - y) ** 2 for p,y in zip(probabilities, labels))
	mse = round(total_squared_error / len(labels), 4)
	return probabilities, mse
