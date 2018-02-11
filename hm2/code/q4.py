import numpy as np

'''
X(numpy array): input data (without intercept)
y(numpy array): input data label
'''

def forwardStepwiseRegModel(X, y):
	# initialization
	[N, p] = X.shape
	M = np.zeros([p + 1, p + 1])
	Gamma = np.zeros([p + 1, p + 1])


	# main loop
	
