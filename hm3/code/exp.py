# experiment framework
import numpy as np
import math
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import ShuffleSplit

from data_proc import parse_data 
from model_lr import model_lr
from model_lr_ridge import model_lr_ridge

from model_lgr import model_lgr
from model_svm import our_svm
import matplotlib.pyplot as plt



path_train = '../data/loan_train.csv'
path_test = '../data/loan_testx.csv'
save_path = '../data/output.csv'

# prepare data for 10-fold cross validation
# return list of tuple:
# [(trainX, trainY, testX, testY)]
def cross_val_data(X,y):
	res = []
	X, y = shuffle(X, y, random_state=0)
	N,p = X.shape
	gap = N/10
	test_err = []
	for i in xrange(10):
		X_test = X[gap * i : gap * (i + 1)]
		y_test = y[gap * i : gap * (i + 1)]
		X_train = list(X[0 : gap * i])
		X_train.extend(list(X[gap * (i + 1) : N]))
		y_train = list(y[0 : gap * i])
		y_train.extend(list(y[gap * (i + 1) : N]))

		X_train = np.reshape(X_train, (-1,p))
		y_train = np.reshape(y_train, (-1, 1))
		res.append((X_train, y_train, X_test, y_test))
	return res


# perform cross validation on ridge regression model
def cross_val(data, model, *arg):
	test_err = []
	for i in xrange(10):
		X_train, y_train, X_test, y_test = data[i]
		mdl = model(X_train, y_train, X_test, y_test, *arg)
		train_error = mdl.train()
		#print "current training error: ", train_error
		_, error = mdl.test()
		test_err.append(error)
	return np.average(test_err), mdl.model



if __name__ == "__main__":
	X, y, names = parse_data(path_train, True)

	# data = cross_val_data(X,y)
	# test_err1, model = cross_val(data, model_lgr)
	# print 'test error of logistic reg model = %.3f' % test_err1

	lgr = model_lgr(X, y, X, y)
	lgr.train()
	#lgr.plot_learning_curve('Learning curve of logistic regression')
	#lgr.plot_coeffi()

	# predict_train = lgr.model.predict_proba(np.hstack((np.ones([X.shape[0], 1]), X)))
	# csv_train = open("../data/output_train.csv", "w")
	# for _, p in predict_train:
	# 	csv_train.write(str(p) + "\n")


	print "calculate test prediction"
	X_test, _ = parse_data(path_test, False)
	X_test = np.hstack((np.ones([X_test.shape[0], 1]), X_test))
	predict = lgr.model.predict_proba(X_test)

	print "calculate 90% prediction interval"
	var = 0.0
	z = 1.645
	N = 10000
	for p0, p1 in predict:
		var += p0 * p1
	var /= N*N
	v1 = np.mean(predict[:,1])
	print var, N
	v2 = math.sqrt(var / N) * z
	a = v1 - v2
	b = v1 + v2

	print "write to csv"
	csvfile = open(save_path, "wb")
	writer = csv.writer(csvfile)
	writer.writerow([str(a)] + [str(b)])
	# csv.write(str(a))
	# csv.write(str(b) + "\n")
	for _, val in predict:
		writer.writerow([str(val)])
		# csv.write(str(val) + "\n")


	# # 2: ridge regression model
	# for lam in [0.1, 1, 10, 100]:

	# 	test_err = cross_val(data, model_lr_ridge, lam)

	# 	print 'test err with lambda = %f is %.3f'% (lam, test_err)


	#test_err_svm = cross_val(data, our_svm)
	#print 'test error for svm:', test_err_svm
