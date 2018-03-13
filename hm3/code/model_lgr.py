# logistic regression model

import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# takes all input features as float values and do linear regression by least squares
# classify as 0 if output < 0.5, classify as 1 otherwise

class model_lgr:
	def __init__(self, trainX, trainY, testX, testY, lam = None):	
		scaler = StandardScaler()
		self.trainX = np.hstack((np.ones([trainX.shape[0], 1]), trainX))	 # N*(p+1) np matrix
		self.trainX = scaler.fit_transform(self.trainX)
		self.trainY = np.reshape(trainY, [-1,])
		self.testX = np.hstack((np.ones([testX.shape[0], 1]), testX))
		self.testX = scaler.transform(self.testX)
		self.testY = np.reshape(testY, [-1,])
		self.model = linear_model.LogisticRegression(C = 1, penalty='l2', tol=1e-4)

	# train linear regression coeff
	def train(self):
		# coeff is p*1 np matrix
		self.model.fit(self.trainX, self.trainY)
		pred = self.model.predict(self.trainX)
		return np.mean(pred != self.trainY)

	# use linear regression coeff, return test error
	def test(self):
		pred = self.model.predict(self.testX)
		#pred_01 = pred
		#pred_01[pred_01 < 0.5] = 0
		#pred_01[pred_01 >= 0.5] = 1
		# test error = sqrt(sum of squares)
		return list(pred), np.mean(pred != self.testY)

	def plot_coeffi(self):
		coef = self.model.coef_
		length = coef.shape[1]
		plt.figure()
		plt.plot(range(1, length), np.abs(np.array(coef[0, 1:length])), marker='o', linewidth=2, color='black', linestyle='dashed')
		plt.ylim(-0.5, 5)
		plt.xlabel('Feature Index')
		plt.ylabel('Absolute value of param')

		plt.show()

	def plot_learning_curve(self, title, ylim=None, cv=None,
							n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

		X = self.trainX
		y = self.trainY
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel("Training examples")
		plt.ylabel("Score")
		train_sizes, train_scores, test_scores = learning_curve(
			self.model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
		train_scores_mean = 1 - np.mean(train_scores, axis=1)
		train_scores_std = np.std(train_scores, axis=1)
		test_scores_mean = 1 - np.mean(test_scores, axis=1)
		test_scores_std = np.std(test_scores, axis=1)
		plt.grid()

		plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
						 train_scores_mean + train_scores_std, alpha=0.1,
						 color="r")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
						 test_scores_mean + test_scores_std, alpha=0.1, color="g")
		plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
				 label="Training error")
		plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
				 label="Cross-validation error")

		plt.legend(loc="best")
		plt.show()