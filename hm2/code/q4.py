import numpy as np
import random
from numpy.linalg import inv
import matplotlib.pyplot as plt


def computeMisclassiError(predict_class, y):
	y = np.reshape(y, [-1, 1])
	return 1 - np.sum(predict_class == y) * 1.0 / len(y)

class StepWiseRegModel:

	fitted_beta = []
	index_pick_order = []

	def findMostEffectiveFeature(self, mat, col_idx, y, residual):
	    min_val = float("inf")
	    #print mat.shape[1]
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
	function: part (d)
			  Fit the data using stepwise regression, and return the parameter

	X(numpy array): input data (without intercept)
	y(numpy array): input data label
	'''
	def forwardStepwiseRegModel(self, X, y, max_iter = -1):
		
		# initialization
		[N, p] = X.shape
		self.fitted_beta = []
		if max_iter == -1:
			max_iter = p

		intercept = np.ones([N, ])
		X = np.hstack((np.reshape(intercept, [N, -1]), X))
		beta = np.zeros(p + 1)
		beta[0] = np.dot(intercept, y) * 1.0 / np.dot(intercept, intercept)

		X_dup = X.copy() # we want to store the original X through the process
		X_used_ind = [0]
		# main loop
		for q in xrange(max_iter):
			last_ind = X_used_ind[-1]
			a = X_dup[:, last_ind] # get the orthogonized vector in last iteration

			for j in xrange(p + 1):
				if j not in X_used_ind:
					X_dup[:, j] -= np.dot(X[:, j], a) * 1.0 / np.dot(a, a) * a

			residual = y - np.matmul(X, beta)

			# find the element that reduces the residual most 
			#print len(X_used_ind)
			new_pick_ind = self.findMostEffectiveFeature(X_dup, X_used_ind, y, residual)
			X_used_ind.append(new_pick_ind)

			curr_X = X[:, X_used_ind]
			new_beta_val = np.matmul(inv(np.matmul(curr_X.T, curr_X)), np.matmul(curr_X.T, np.reshape(y, [N, -1])))
			new_beta_mapback = np.zeros([p + 1, ]) 
			new_beta_mapback[X_used_ind] = new_beta_val
		
			self.fitted_beta.append(new_beta_mapback)

		self.index_pick_order = X_used_ind
		return self.fitted_beta


	def predict(self, X, use_step = -1):
		[N, p] = X.shape
		intercept = np.ones([N, ])
		X = np.hstack((np.reshape(intercept, [N, -1]), X))
		beta_pick = self.fitted_beta[use_step]
		regress_result = np.matmul(X, np.reshape(beta_pick, [p + 1, -1]))
		predict_class = (regress_result >= 0.5) * 1
		return predict_class

	def predictAllStepAndComputeStats(self, X, y, depict = False, data_type = 'train'):
		[N, p] = X.shape
		y = np.reshape(y, [-1, 1])
		intercept = np.ones([N, ])
		X = np.hstack((np.reshape(intercept, [N, -1]), X))
		RSS_list = []
		classification_error = []
		for i in range(len(self.fitted_beta)):
			beta_pick = self.fitted_beta[i]
			regress_result = np.matmul(X, np.reshape(beta_pick, [p + 1, -1]))
			predict_class = (regress_result >= 0.5) * 1
			RSS_list.append(np.sum( (regress_result - y) ** 2))
			classification_error.append(computeMisclassiError(predict_class, y))


		if depict:
			plt.figure(1)
			plt.plot(range(len(RSS_list)), RSS_list)
			plt.xlabel('Iteration Steps')
			plt.ylabel('RSS')
			plt.savefig('learn_rss_%s.png'%(data_type))
			plt.show()

			plt.figure(2)
			plt.plot(range(len(classification_error)), classification_error)
			plt.xlabel('Iteration Steps')
			plt.ylabel('Misclassification Error on training set')
			plt.savefig('learn_error_%s.png'%(data_type))
			plt.show()

		return RSS_list, classification_error

	def perform10Fold(self, X, y):
		# first split data into 10 folds
		print "Perform 10 fold cross validation..."

		[N, p] = X.shape
		gap = N / 10
		classification_error = np.zeros([10, p])

		for i in xrange(10):
			X_valid = X[gap * i : gap * (i + 1)]
			y_valid = y[gap * i : gap * (i + 1)]
			X_train = list(X[0 : gap * i])
			X_train.extend(list(X[gap * (i + 1) : N]))
			X_train = np.array(X_train)
			y_train = list(y[0 : gap * i])
			y_train.extend(list(y[gap * (i + 1) : N]))
			y_train = np.array(y_train)

			#print X.shape, y.shape, X_train.shape, y_train.shape
			
			self.forwardStepwiseRegModel(X_train, y_train)
			
			_, cur_error = self.predictAllStepAndComputeStats(X_valid, y_valid)

			classification_error[i, :] = np.array(cur_error)
			print "Fold ", i + 1, " processed!"

		
		mean_classification_error = list(np.mean(classification_error, axis = 0))
		#mean_classification_error = list(classification_error[1, :])
		std_error = list(np.std(classification_error, axis = 0) / np.sqrt(10))
		#print std_error

		plt.figure(1)
		plt.plot(range(len(mean_classification_error)), mean_classification_error)
		plt.xlabel('Iteration Steps')
		plt.ylabel('Misclassication Error of 10 Fold')
		plt.savefig('10fold_error.png')
		plt.show()

		plt.figure(2)
		plt.plot(range(len(std_error)), std_error)
		plt.xlabel('Iteration Steps')
		plt.ylabel('Standard Error of 10 fold')
		plt.savefig('10fold_std.png')
		plt.show()

		plt.figure(3)
		plt.errorbar(range(len(std_error)), mean_classification_error, std_error, marker='^')
		plt.xlabel('Iteration Steps')
		plt.ylabel('Mean and Standard Error of 10 fold')
		plt.savefig('10fold_all.png')
		plt.show()

		print "Congrats! 10 fold cross validation finished successfully!"

	def plot


def loadSpamData(data_path, index_path):
	all_data = np.loadtxt(data_path, delimiter =' ')
	y = all_data[:, -1]
	X = all_data[:, 0 : all_data.shape[1] - 1]
	ind = np.loadtxt(index_path, delimiter =' ')
	#print X[1, -3:]
	X[:, -3:] = np.log(X[:, -3:])
	X[:, : -3] = (X[:, : -3] == 0) * 1

	ind_train = []
	ind_test = []
	for i in xrange(len(ind)):
		if ind[i] == 0:
			ind_train.append(i)
		else:
			ind_test.append(i)

	X_train = X[ind_train, :]
	y_train = y[ind_train]
	X_test = X[ind_test, :]
	y_test = y[ind_test]
	return X_train, y_train, X_test, y_test


if __name__ == '__main__':
	# X = np.array([[1,0,0], [0,1,0],[0,0,1]])
	# y = np.array([3, 6, 9])

	# model = StepWiseRegModel()

	# model.forwardStepwiseRegModel(X, y)
	# print model.fitted_beta
	# print model.predict(X)

	X_train, y_train, X_test, y_test = loadSpamData('./spambase/spam.data', './spambase/spam.index')

	model = StepWiseRegModel()

	
	######  train on the spam training set#######
	
	model.forwardStepwiseRegModel(X_train, y_train)
	predict_class =  model.predict(X_train)
	
	##### plot learning curve #####
	
	model.predictAllStepAndComputeStats(X_train, y_train, depict = True)
	print computeMisclassiError(predict_class, y_train)


	######  10 fold cross validation #######
	
	model.perform10Fold(X_train, y_train)

	######  predict on the testing set and plot the prediction curve #######
	#model.predictAllStepAndComputeStats(X_test, y_test, depict = True, data_type = 'test')











