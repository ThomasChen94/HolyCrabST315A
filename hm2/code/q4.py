import numpy as np
import random


# def findMostEffectiveFeature(X_dup, X_used_ind, y, residual):
# 	return random.randint(1, 3)


def findMostEffectiveFeature(mat, col_idx, y, residual):
    min_val = float("inf")
    for i in range(mat.shape[1]):
    	if i not in col_idx:
	        vec = mat[:,i]
	        beta = float(np.dot(vec, y)) / np.dot(vec, vec)
	        #print vec
	        val = np.linalg.norm(residual - beta * vec)
	        if val < min_val:
	            min_idx = i
	            min_val = val
    return min_idx


'''
X(numpy array): input data (without intercept)
y(numpy array): input data label
'''
def forwardStepwiseRegModel(X, y):
	
	# initialization
	[N, p] = X.shape
	intercept = np.ones([N, ])
	X = np.hstack((np.reshape(intercept, [N, -1]), X))
	beta = np.zeros(p + 1)
	beta[0] = np.dot(intercept, y) * 1.0 / np.dot(intercept, intercept)


	M = np.zeros([p + 1, p + 1])
	Gamma = np.zeros([p + 1, p + 1])
	gamma_list = []
	X_dup = X.copy() # we want to store the original X through the process
	X_used_ind = [0]
	# main loop
	for q in xrange(p):
		last_ind = X_used_ind[-1]
		a = X_dup[:, last_ind] # get the orthogonized vector in last iteration
	
		for j in xrange(p):
			if j not in X_used_ind:
				X_dup[:, j] -= np.dot(X[:, j], a) * 1.0 / np.dot(a, a) * a
				print "gamma: ", np.dot(X[:, j], a) * 1.0 / np.dot(a, a) * a
		residual = y - np.matmul(X, beta)
		#print residual, "    beta: ", beta
		# find the element that reduces the residual most 
		new_pick_ind = findMostEffectiveFeature(X_dup, X_used_ind, y, residual)
		zp = X_dup[:, new_pick_ind] # the column we process at this iteration
		print zp, "\n"
		beta[new_pick_ind] = np.dot(zp, y) * 1.0 / np.dot(zp, zp)
		X_used_ind.append(new_pick_ind)


	return beta



if __name__ == '__main__':
	X = np.array([[1,0,0], [0,1,0],[0,0,1]])
	y = np.array([3, 6, 9])
	print forwardStepwiseRegModel(X, y)













