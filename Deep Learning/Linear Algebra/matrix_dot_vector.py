def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
    if len(a[0])!=len(b):
        return -1
    result = []
    for row in a:
        row_sum = 0
        for val_a, val_b in zip(row,b):
            row_sum+=val_a*val_b
        result.append(row_sum)
    return result
