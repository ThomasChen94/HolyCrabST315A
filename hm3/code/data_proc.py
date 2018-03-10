import numpy as np 
import csv


numFeature = 30

# process input data
# if train, output (features (n*30), labels (n*1), name of features)
# if test, output (features, name of features)
def parse_data(path, train = True):
	X = None # n*30 features
	y = None # n*1 labels
	names = None # 1*30 name of features

	with open(path, 'r') as csvfile:
		reader = csv.reader(csvfile)
		cnt = 0
		featureStartIdx = 1 if train else 0

		## read in lines from csv file
		for row in reader:
			if cnt == 0:
				names = row[1:]
				# print "names:", names, len(names)
			else:
				if X is None:
					X = np.reshape(row[featureStartIdx:], (1, numFeature))
					if train:
						y = [row[0]]
				else:
					X = np.vstack((X, np.reshape(row[featureStartIdx:], (1, numFeature))))
					if train:
						y.append(row[0])
				# print X, X.shape
				# print y, len(y)
				# print
			cnt += 1
			# if cnt == 10:break
			
	## processing for different features
	int_list = [0, 10, 12, 14, 23,24]
	float_list = [1, 2, 4,5,6,7,9, 13,16,17,19,20,22, 25,26,27, 28, 29]
	special_list = [3, 8, 11, 15, 18,21]

	for row in range(X.shape[0]): # process special list
		m3 = {'a':-1, 'b':1}
		if X[row,3] in m3:
			X[row,3] = m3[X[row,3]]
		else:
			print 'not a or b', X[row,3]

		m8 = {' 3 yrs':-1, ' 5 yrs':1}
		if X[row,8] in m8:
			X[row,8] = m8[X[row,8]]
		else:
			print "not 3 or 5 years", X[row,8]

		m11 = {'NA':0, '< 1':-1, '10+':15} ## careful for this -1! doesn't make sense mathematically

		if X[row,11] in m11:
			X[row,11] = m11[X[row,11]]
		else:
			X[row, 11] = int(X[row, 11])

		m15 = {'checked':0, 'partial':1, 'unchecked':2}
		X[row,15] = m15[X[row,15]]

		m18 = {'debt':0, 'renovation':1, 'cc':2, 'business':3, 'home': 4, 
			   'transport':5, 'moving':6, 'medical':7, 'boat':8,
			   'holiday': 9, 'other':10, 'solar':11, 'event':12, 'educ':13}
		if X[row,18] in m18:
			X[row,18] = m18[X[row,18]]
		else:
			print 'm18 not found', X[row, 18]

		m21 = {'q1':1, 'q2':2, 'q3':3, 'q4':4, 'q5':5, 'q6':6, 'q7':7}
		X[row,21] = m21[X[row,21]]

	X = X.astype(float)
	print 'dim of features:', X.shape
	if train:
		y = np.reshape(y, (-1,1))
		y = y.astype(float)
		print 'dim of labels:', y.shape

	if train:
		return (X, y, names)
	else:
		return (X, names)






