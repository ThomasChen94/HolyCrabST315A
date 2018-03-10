import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class our_svm:
	def __init__(self, trainX, trainY, testX, testY):
		scaler = StandardScaler()
		self.trainX = np.hstack((np.ones([trainX.shape[0], 1]), trainX))	 # N*(p+1) np matrix
		self.trainX = scaler.fit_transform(self.trainX)
		self.trainY = trainY.flat
		self.testX = np.hstack((np.ones([testX.shape[0], 1]), testX))
		self.testX = scaler.transform(self.testX)
		self.testY = testY.flat
		self.svc = SVC(C=8, gamma=0.01, probability=False, verbose=False, kernel="rbf")

	def train(self):
		self.svc.fit(self.trainX, self.trainY)

	def test(self):
		pred = self.svc.predict(self.testX)
		pred[pred < 0.5] = 0
		pred[pred >= 0.5] = 1
		return np.mean(pred != self.testY)
