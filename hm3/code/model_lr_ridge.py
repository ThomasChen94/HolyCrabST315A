# ridge regression model

import numpy as np

# linear regression + ridge penalty
# classify as 0 if output < 0.5, classify as 1 otherwise

class model_lr_ridge:
	def __init__(self, trainX, trainY, testX, testY, lam):	
		self.trainX = np.hstack((np.ones([trainX.shape[0], 1]), trainX))	 # N*(p+1) np matrix
		self.trainY = trainY	# N*1 np matrix
		self.testX = np.hstack((np.ones([testX.shape[0], 1]), testX))
		self.testY = testY
		self.lam = lam

	# train linear regression coeff
	def train(self):
		# coeff is p*1 np matrix
		mat = np.linalg.inv(np.matmul(self.trainX.T, self.trainX) + 
								  self.lam * np.identity(self.trainX.shape[1]))
		vec = np.dot(self.trainX.T, self.trainY)
		self.beta = np.dot(mat, vec)

	# use linear regression coeff, return test error
	def test(self):
		pred = np.dot(self.testX, self.beta)
		pred[pred < 0.5] = 0
		pred[pred >= 0.5] = 1
		# test error = sqrt(sum of squares)
		return np.mean(pred != self.testY)
