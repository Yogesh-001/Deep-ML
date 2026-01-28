def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    """
    Transpose a 2D matrix by swapping rows and columns.
    
    Args:
        a: A 2D matrix of shape (m, n)
    
    Returns:
        The transposed matrix of shape (n, m)
    """
    # Your code here
    # Double transpose: ( A.T ).T = A
    # Sum:(A+B).T=A.T+B.T
    # Scalar multiplication: ( cA ).T = cA.T
    # Product: (A B).T = B.T * A.T (note the reversed order)
    # Symmetric matrix: A = A.T means A is symmetric
    result = []
    for i in range(len(a[0])):
        res1 = []
        for row in a:
            res1.append(row[i])
        result.append(res1)
    return result
