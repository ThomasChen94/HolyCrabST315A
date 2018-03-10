# linear regression model

import numpy as np

# takes all input features as float values and do linear regression by least squares
# classify as 0 if output < 0.5, classify as 1 otherwise

class model_lr:
	def __init__(self, trainX, trainY, testX, testY, lam = None):	
		self.trainX = np.hstack((np.ones([trainX.shape[0], 1]), trainX))	 # N*(p+1) np matrix
		self.trainY = trainY	# N*1 np matrix
		self.testX = np.hstack((np.ones([testX.shape[0], 1]), testX))
		self.testY = testY

	# train linear regression coeff
	def train(self):
		# coeff is p*1 np matrix
		self.beta = np.linalg.lstsq(self.trainX, self.trainY, rcond = None)[0]

	# use linear regression coeff, return test error
	def test(self):
		pred = np.dot(self.testX, self.beta)
		pred[pred < 0.5] = 0
		pred[pred >= 0.5] = 1
		# test error = sqrt(sum of squares)
		return np.mean(pred != self.testY)
