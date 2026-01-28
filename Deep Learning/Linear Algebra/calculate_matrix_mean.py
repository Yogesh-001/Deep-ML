import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    mean_len = len(matrix[0])
    matrix_len = len(matrix)
    means=[]
    if mode == "row":
        for row in matrix:
            means.append(np.sum(row) / mean_len)
    else:
        for i in range(mean_len):
            col_sum=0
            for j in range(matrix_len):
                col_sum+=matrix[j][i]
            
            means.append(col_sum / matrix_len)
	return means
