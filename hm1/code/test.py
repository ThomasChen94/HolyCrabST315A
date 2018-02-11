import os
import numpy as np
import pickle

if os.path.exists('data.txt'):
	f = open('data.txt', 'rb')
	data = pickle.load(f)
	print data
	f.close()
else:
	a = [[1,2,3], [[1,2], [2,3]]]
	f = open('data.txt', 'wb')
	pickle.dump(a, f)



