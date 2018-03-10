# logistic regression model

import numpy as np
from sklearn import linear_model


# takes all input features as float values and do linear regression by least squares
# classify as 0 if output < 0.5, classify as 1 otherwise

class model_lgr:
	def __init__(self, trainX, trainY, testX, testY, lam = None):	
		
		self.trainX = np.hstack((np.ones([trainX.shape[0], 1]), trainX))	 # N*(p+1) np matrix
		self.trainY = trainY.flat	# N*1 np matrix
		self.testX = np.hstack((np.ones([testX.shape[0], 1]), testX))
		self.testY = testY.flat
		self.model = linear_model.LogisticRegression()

	# train linear regression coeff
	def train(self):
		# coeff is p*1 np matrix
		self.model.fit(self.trainX, self.trainY)

	# use linear regression coeff, return test error
	def test(self):
		pred = self.model.predict(self.testX)
		#pred_01 = pred
		#pred_01[pred_01 < 0.5] = 0
		#pred_01[pred_01 >= 0.5] = 1
		# test error = sqrt(sum of squares)
		return list(pred), np.mean(pred != self.testY)
