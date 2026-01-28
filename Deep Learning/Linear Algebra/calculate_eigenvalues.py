import numpy as np
def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    # λ**2-tr(A)λ+det(A)=0
    trace = matrix[0][0] + matrix[1][1]
    det = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    lambda1 = (trace + np.sqrt(trace**2 - 4 * det)) / 2
    lambda2 = (trace - np.sqrt(trace**2 - 4 * det)) / 2
    eigenvalues = sorted([lambda1, lambda2],reverse=True)
	return eigenvalues
