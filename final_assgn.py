import numpy as np

#Creating a class named matrix
class matrix:
	# Creating a numpy attribute array_2d which will be 2-dimensions
	array_2d = np.array([])
	
	#initialising a method
	def __init__(self, filename=None, array=None):
		#self.filename = filename
		if filename != None:
			self.array_2d = self.load_from_csv(filename)

		if array is not None:
			self.array_2d = array

	# Created a method load_from_csv
	def load_from_csv(self, filename:str):
		f = open(filename, 'r')
		lines = f.readlines()
		my_list = []
		for d in lines:
			value = list(map(float, d.split(',')))
			my_list.append(value)

			# Storing data in array_2d attribute
			self.array_2d = np.array(my_list)
			f.close()
			return self.array_2d

	# Standardization method for the given equation
	def standardise(self):
		array = self.array_2d.copy()
		self.array_2d = array - np.mean(array, 0)
		self.array_2d = self.array_2d/(np.max(array, 0) - np.min(array, 0))

	# get_distance method will calculate the distance using weights and beta
	def get_distance(self, other_matrix, weights, beta):
		if self.array_2d.shape[0] == 1:
			distance = np.zeros((other_matrix.array_2d.shape[0], 1))
			for i in range(other_matrix.array_2d.shape[0]):
				param1 = self.array_2d.reshape(1, -1)
				param2 = other_matrix.array_2d[i,:].reshape(1, -1)
				difference = np.square(param1 - param2)
				weigh = np.power(weights, beta).reshape(-1, 1)
				distance_all = np.sum(np.dot(weigh, difference))
				distance[i, :] = distance_all
			return distance

	def get_count_frequency(self):
		if self.array_2d.shape[1] == 1:
			dictionary = {}
			for i in self.array_2d:
				if i[0] in dictionary:
					dictionary[i[0]] += 1
				else:
					dictionary[i[0]] = 0
			return dictionary

#Creation functions from now

def get_initial_weights(m):
	array = np.random.rand(m)
	array = array/np.sum(array)
	return array

# get_centroid will find the mean to get new centroid
def get_centroids(m: matrix, S: np.ndarray, K: int):
	m.standardise()
	centroid = np.zeros((K, m.array_2d.shape[1]))
	for row in range(K):
		c_row = []
		for m_rix in range(S.shape[0]):
			c_row.append(m.array_2d[m_rix, :])
		centroid[row,:] = sum(c_row)/len(c_row)

	return matrix(array=centroid)

def get_groups(data_matrix, K, beta):
	data_matrix.standardise()
	w1,w2 = data_matrix.array_2d.shape
	wt = get_initial_weights(w2)
	centroid = matrix()
	S = np.zeros((w1,1))
	row = np.random.choice(w1, K)
	centroid.array_2d = data_matrix.array_2d[row,:].copy()

	while True:
		count = 0
		for i in range(w1):
			m_trix = matrix(array = data_matrix.array_2d[i,:].reshape(1,-1))
			distance = m_trix.get_distance(centroid, wt, beta)
			new_centroid = np.argmin(distance)
			if S[i,:] != new_centroid:
				count = count + 1
				S[i,:] = new_centroid

		if count == 0:
			return matrix(array = S)

		centroid = get_centroids(m,S,K)
		weigh = get_new_weights(m, centroid, S, beta)

# Function to calculate dispersion of weights
def get_new_weights(m, centroid, S, beta):
	w1,w2 = data_matrix.array_2d.shape
	wt = np.zeros((1,w2))
	K,_ = centroid.array_2d.shape
	dispersion = np.zeros((1,w2))
	for row in range(K):
		for i in range(w1):
			if(S[i,:] == row):
				dispersion = dispersion + np.square(m.array_2d[i,:] - centroid.array_2d[row,:])

	for j in range(w2):
		if dispersion[0,j] == 0:
			wt[0,j] = 0
		else:
			ctr = 0
			for i in range(w2):
				ctr = ctr + pow(dispersion[0,j]/disp[0,i], 1/(beta-1))
			wt[0,j] = ctr

	return wt 

def run_test():
    m = matrix('Data.csv')
    for k in range(2,5):
        for beta in range(11,25):
            S = get_groups(m, k, beta/10)
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))
            


run_test()