def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
    n = len(vectors[0])
    m = len(vectors)

    means = [sum(feature) / n for feature in vectors]
    covariance_matrix = [[0.0 for _ in range(m)]for _ in range(m)]

    for i in range(m):
        for j in range(m):
            sum_vectors = 0
            for k in range(n):
                deviation_i = vectors[i][k] - means[i]
                deviation_j = vectors[j][k] - means[j]
                sum_vectors += deviation_i * deviation_j
            covariance_matrix[i][j] = sum_vectors / (n-1)
	return covariance_matrix
