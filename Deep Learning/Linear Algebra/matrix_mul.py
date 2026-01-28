def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:
    if len(a[0]) != len(b):
        return -1
    rows_a = len(a)
    col_b = len(b[0])
    rows_b = len(b)
    c = [[0 for _ in range(col_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(col_b):
            for k in range(rows_b):
                c[i][j]+= a[i][k] * b[k][j]
    
	return c
