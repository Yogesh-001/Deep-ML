import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (μ) of the distribution.
	:param std_dev: The standard deviation (σ) of the distribution.
	"""
	# Your code here
	z_score = (x-mean) / std_dev
    coefficient = 1 / (std_dev * math.sqrt(2 * math.pi))
    exponent = math.exp(-0.5 * z_score**2)

    val = coefficient * exponent
	return round(val,5)
