import numpy as np
import os
import matplotlib.pyplot as plt  
from sklearn import neighbors
from sklearn import linear_model
import pickle
from matplotlib.colors import ListedColormap

NUM_CENTROIDS_PER_CLASS = 10


def generateSampleData(centroids, sampleSize, noiseVar):


	# First generate the centroid matrix for both classes

	centroidIdx = np.random.randint(0, NUM_CENTROIDS_PER_CLASS, [sampleSize, ])

	# second generate data
	
	data = [np.random.multivariate_normal(centroids[centroidIdx[i]], noiseVar) \
								for i in range(sampleSize)]

	return data


def perfromKNN(trainData, trainLabel, testData, testLabel, numNeighbors):

	knn = neighbors.KNeighborsClassifier(numNeighbors) # get the classifier  	
	knn.fit(list(trainData), list(trainLabel)) # store the feature value
	predict = []
	for i in range(len(testLabel)):
		tiny_data = np.reshape(testData[i], [1, -1])
		predict.append(knn.predict(tiny_data))
	#predict = [knn.predict(testData[i]) for i in range(len(testLabel))]
	check = [1 if predict[i] == testLabel[i] else 0 for i in range(len(testLabel))]
	numCorrect = sum(check)

	return numCorrect * 1.0 / len(testLabel)

def perfromLC(trainData, trainLabel, testData, testLabel):

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(trainData, trainLabel)

	# Make predictions using the testing set
	predict_regr = regr.predict(testData)
	predict = [1 if val > 0.5 else 0 for val in predict_regr]
	check = [1 if predict[i] == testLabel[i] else 0 for i in range(len(testLabel))]
	numCorrect = sum(check)

	return numCorrect * 1.0 / len(testLabel)

def perform10Fold(trainData, trainLabel, numNeighbors):
	knn = neighbors.KNeighborsClassifier(numNeighbors) # get the classifier
	errorSum = 0
	gap = len(trainData) / 10
	validationError = []
	for i in range(10):
		validationData = trainData[gap * i : gap * (i + 1)]
		validationLabel = trainLabel[gap * i : gap * (i + 1)]
		realTrainData = list(trainData[0 : gap * i])
		realTrainData.extend(list(trainData[gap * (i + 1) : len(trainData)]))
		realTrainLabel = list(trainLabel[0 : gap * i])
		realTrainLabel.extend(list(trainLabel[gap * (i + 1) : len(trainLabel)]))
		knn.fit(list(realTrainData), list(realTrainLabel)) # store the feature value
		predict = []
		for i in range(len(validationLabel)):
			tiny_data = np.reshape(validationData[i], [1, -1])
			predict.append(knn.predict(tiny_data))

		checkWrong = [1 if predict[i] != validationLabel[i] else 0 for i in range(len(validationLabel))]
		numWrong = sum(checkWrong)
		validationError.append(numWrong * 1.0 / gap)



	return np.mean(np.array(validationError)), np.std(np.array(validationError))

def prepare_data(path):
	if os.path.exists(path):
		f = open(path, 'rb')
		data = pickle.load(f)
		f.close()
		return data
	else:

		trainSampleSize = 100
		testSampleSize = 10000
		noiseVar = [[0.2, 0], [0, 0.2]]

		centroidVar = [[1, 0], [0, 1]]
		mu1 = [1, 0]
		mu2 = [0, 1]

		centroids1 = np.random.multivariate_normal(mu1, centroidVar, [10,])
		centroids2 = np.random.multivariate_normal(mu2, centroidVar, [10,])

		trainData1 = generateSampleData(centroids1, trainSampleSize, noiseVar)
		trainData2 = generateSampleData(centroids2, trainSampleSize, noiseVar)
		trainData  = np.concatenate((trainData1, trainData2), axis = 0)
		trainLabel = list(np.zeros(trainSampleSize))
		trainLabel.extend(list(np.ones(trainSampleSize)))

		testData1 = generateSampleData(centroids1, testSampleSize, noiseVar)
		testData2 = generateSampleData(centroids2, testSampleSize, noiseVar)
		testData  = list(np.concatenate((testData1, testData2), axis = 0))
		testLabel = list(np.zeros(trainSampleSize))
		testLabel.extend(list(np.ones(trainSampleSize)))

		data = [trainData, trainLabel, testData, testLabel, centroids1, centroids2]

		f = open(path, 'wb')
		pickle.dump(data, f)
		f.close()

		return data


def bayesianDecision(centroids1, centroids2, point, var):
	p1 = 0
	p2 = 0
	denominator = 2.0 * var ** 2
	for c in centroids1:
		p1 += np.exp( -np.sum(np.square(point - c)) / denominator)
	for c in centroids2:
		p2 += np.exp( -np.sum(np.square(point - c)) / denominator)
	if p1 > p2:
		return 0
	else:
		return 1


def drawDecisionBound(trainData, trainLabel, numNeighbors, centroids1, centroids2):
	knn = neighbors.KNeighborsClassifier(numNeighbors) # get the classifier
	knn.fit(list(trainData), list(trainLabel)) # store the feature value
	# Plot the decision boundary. For that, we will assign a color to each
	# point in the mesh [x_min, x_max]x[y_min, y_max].
	h = 0.2
	x_min, x_max = trainData[:, 0].min() - 1, trainData[:, 0].max() + 1
	y_min, y_max = trainData[:, 1].min() - 1, trainData[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						 np.arange(y_min, y_max, h))
	Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
	bayes = np.array([bayesianDecision(centroids1, centroids2, point, 0.2) for point in list(np.c_[xx.ravel(), yy.ravel()])])

	# Create color maps
	cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
	cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	bayes = bayes.reshape(xx.shape)
	plt.figure()
	#plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
	plt.contour(xx, yy, Z, [0.5,], color = 'r')
	plt.contour(xx, yy, bayes, [0.5,], color = 'g')

	# Plot also the training points
	plt.scatter(trainData[:, 0], trainData[:, 1], c=trainLabel, cmap=cmap_bold,
				edgecolor='k', s=20)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()


[trainData, trainLabel, testData, testLabel, centroids1, centroids2] = prepare_data('data.txt')

kValue = [1, 3, 5, 9, 15, 25, 45, 83, 151]
degFree = [200 / kValue[i] for i in range(len(kValue))]

# valueRange = range(len(kValue))
# trainError = [1 - perfromKNN(trainData, trainLabel, trainData, trainLabel, kValue[i]) for i in range(len(kValue))]
# testError = [1 - perfromKNN(trainData, trainLabel, testData, testLabel, kValue[i]) for i in range(len(kValue))]
#
#
# print trainError[::-1]
# print testError[::-1]
#
#

# trainErrorLC = 1 - perfromLC(trainData, trainLabel, trainData, trainLabel)
# testErrorLC = 1 - perfromLC(trainData, trainLabel, testData, testLabel)
#
#
# plt.plot(degFree, trainError, 'r-o')
# plt.plot(degFree, testError, 'b-o')
# plt.plot(degFree[3], trainErrorLC, 'r--s')
# plt.plot(degFree[3], testErrorLC, 'b--s')
#
# plt.show()

#
# validationError = []
# validationStd = []
# for i in range(len(kValue)):
# 	tmpError, tmpStd = perform10Fold(trainData, trainLabel, kValue[i])
# 	validationError.append(tmpError)
# 	validationStd.append(tmpStd)
#
#
# plt.plot(degFree, validationError, 'r-o')
# plt.plot(degFree, validationStd, 'b-o')
# plt.show()
#
# print validationError
# print validationStd


# print validationError

drawDecisionBound(trainData, trainLabel, kValue[6], centroids1, centroids2)
# testError = perfromLC(trainData, trainLabel, testData, testLabel)
# print testError