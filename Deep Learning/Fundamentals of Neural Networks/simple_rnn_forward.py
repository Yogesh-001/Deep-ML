import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	# Your code here
	wx = np.array(Wx)
	wh = np.array(Wh)
	b = np.array(b)
	h = np.array(initial_hidden_state)

	for x in input_sequence:
		x = np.array(x)
		h = np.tanh(np.dot(wx,x) + np.dot(wh,h) + b)
	
	final_hidden_state = np.round(h,4).tolist()
	return final_hidden_state
