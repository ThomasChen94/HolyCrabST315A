import copy

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


TRAIN_DATA_FILE = "../data/q7.train"
TEST_DATA_FILE = "../data/q7.test"


class Question_7():

	class Data():
		def __init__(self):
			self.x = None
			self.y = None
			
	def __init__(self, train_file, test_file):
		self._train = self.Data()
		self._test = self.Data()
		self.filtered_data = None
		self.load_data(train_file, test_file)
	
	def select_data(self, data):
		new_data = []
		for record in data:
			if record[0] in [3, 5, 8]:
				new_data.append(record)
		return np.array(new_data)

	def load_data(self, train_file, test_file):
		train_data = np.loadtxt(train_file)
		test_data = np.loadtxt(test_file)
		_train_data = self.select_data(train_data)
		_test_data = self.select_data(test_data)
		mean = np.mean(_train_data[:,1:], axis=0)
		self._train.x = _train_data[:,1:] - mean
		self._train.y = _train_data[:,0]
		self._test.x = _test_data[:,1:] - mean
		self._test.y = _test_data[:,0]

	def get_error(self, true_y, predict_y):
		return float(sum(true_y != predict_y)) / len(true_y)

	def model_fit_predict(self, model, train, test):
		fitted_model = model.fit(train.x, train.y)
		predict_train = fitted_model.predict(train.x)
		predict_test = fitted_model.predict(test.x)
		train_error = self.get_error(train.y, predict_train)
		test_error = self.get_error(test.y, predict_test)
		return train_error, test_error

	def LDA_fit_predict(self, *data):
		lda = LinearDiscriminantAnalysis()
		return self.model_fit_predict(lda, *data)

	def logistic_fit_predict(self, *data):
		lr = LogisticRegression(
			solver="lbfgs",
			max_iter=1000,
			multi_class="multinomial",
			verbose=0)
		return self.model_fit_predict(lr, *data)

	def sub_q_a(self):
		print("a:", self.LDA_fit_predict(self._train, self._test))

	def sub_q_b(self):
		pca = PCA(n_components=30)
		pca_model = pca.fit(self._train.x)
		train, test = self.Data(), self.Data()
		train.x = pca_model.transform(self._train.x)
		test.x = pca_model.transform(self._test.x)
		train.y = self._train.y
		test.y = self._test.y
		print("b:", self.LDA_fit_predict(train, test))

	def sub_q_c(self):

		def get_components_from_n(number, old_x):
			pca = PCA(n_components=10)
			selector = self._train.y == number
			x = self._train.x[selector, :]
			new_x = pca.fit(x).transform(old_x)
			return new_x

		train, test = self.Data(), self.Data()
		train.y = self._train.y
		test.y = self._test.y
		train.x = np.hstack((
			get_components_from_n(3, self._train.x),
			get_components_from_n(5, self._train.x),
			get_components_from_n(8, self._train.x),
		))
		test.x = np.hstack((
			get_components_from_n(3, self._test.x),
			get_components_from_n(5, self._test.x),
			get_components_from_n(8, self._test.x),
		))
		print("c:", self.LDA_fit_predict(train, test))

	def average_filter(self, old_x):
		new_x = np.zeros(old_x.shape)
		for j in range(4):
			for m in range(4):
				i = (j + m * 16) * 4
				s = (np.sum(old_x[:,i:i+4], axis=1)
					+ np.sum(old_x[:,i+16:i+20], axis=1)
					+ np.sum(old_x[:,i+32:i+36], axis=1)
					+ np.sum(old_x[:,i+48:i+52], axis=1))
				vec_s = np.reshape(s, (-1, 1)) / 16
				new_x[:,i:i+4] = vec_s
				new_x[:,i+16:i+20] = vec_s
				new_x[:,i+32:i+36] = vec_s
				new_x[:,i+48:i+52] = vec_s
		return new_x

	def filter_data(self):
		if self.filtered_data is None:
			train, test = self.Data(), self.Data()
			train.y = self._train.y
			test.y = self._test.y
			train.x = self.average_filter(self._train.x)
			test.x = self.average_filter(self._test.x)
			self.filtered_data = [train, test]

	def sub_q_d(self):
		self.filter_data()
		print("d:", self.LDA_fit_predict(*self.filtered_data))

	def sub_q_e(self):
		self.filter_data()
		print("e:", self.logistic_fit_predict(*self.filtered_data))

	def export_filtered_data(self):
		self.filter_data()
		train, test = self.filtered_data
		np.savetxt("train.x", train.x, delimiter=",", fmt="%4.4f")
		np.savetxt("train.y", train.y, delimiter=",", fmt="%4.4f")
		np.savetxt("test.x", test.x, delimiter=",", fmt="%4.4f")
		np.savetxt("test.y", test.y, delimiter=",", fmt="%4.4f")


if __name__ == '__main__':
	model = Question_7(TRAIN_DATA_FILE, TEST_DATA_FILE)
	model.sub_q_a()
	model.sub_q_b()
	model.sub_q_c()
	model.sub_q_d()
	# model.sub_q_e()
	# using R code for question e
	model.export_filtered_data()
