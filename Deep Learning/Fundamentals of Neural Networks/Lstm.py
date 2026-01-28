import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))
	
	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		h = initial_hidden_state
		c = initial_cell_state

		h_states = []

		for xt in x:

			xt = xt.reshape(-1,1)

			concat = np.vstack((h, xt))

			ft = self.sigmoid(np.dot(self.Wf, concat) + self.bf)
			it = self.sigmoid(np.dot(self.Wi, concat) + self.bi)
			c_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
			ot = self.sigmoid(np.dot(self.Wo, concat) + self.bo)

			c = ft * c + it * c_tilde
			h = ot * np.tanh(c)

			h_states.append(h)

		return h_states, h, c
