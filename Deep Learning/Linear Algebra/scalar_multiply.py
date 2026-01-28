def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    # result = []
    # for row in matrix:
    #     res1= []
    #     for num in row:
    #         res1.append(num * scalar)
    #     result.append(res1)
    
    result = [[num * scalar for num in row] for row in matrix]
	return result
