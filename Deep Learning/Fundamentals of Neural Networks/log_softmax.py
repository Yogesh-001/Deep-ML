import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	# Your code here
	max_scores = np.array(max(scores))

	log_scores = scores - max_scores

	log_sum_exp = np.log(np.sum(np.exp(log_scores)))

	log_softmax = log_scores - log_sum_exp

	return log_softmax
