import numpy as np

# This helper function will be called by forwardStepwiseRegression function.
# The functionality is as 4(b)
#
def findMostEffectiveFeature(mat, col_idx, y, residual):
    min_val = float("inf")
    for i in col_idx:
        vec = mat[:,i]
        beta = float(np.dot(vec, y)) / np.dot(vec, vec)
        val = np.linalg.norm(residual - beta * vec)
        if val < min_val:
            min_idx = i
            min_val = val
    return min_idx

## testing
mat = np.array([[1, 2, 3], [4, 5, 6]])
col_idx = [2, 0]
y = np.array([1,2])
residual = np.array([1,2])
print findMostEffectiveFeature(mat, col_idx, y, residual)

