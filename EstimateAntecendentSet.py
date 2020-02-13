import numpy as np
from scipy.optimize import curve_fit

class AntecedentEstimator(object):
	def __init__(self, x_train, partition_matrix, mf_shape='gauss'):
		self.xtrain=x_train
		self.partition_matrix=partition_matrix
		self._info_for_simplification = None
				
		
	def determineMF(self,x_train,partition_matrix,mf_shape='gauss', merge_threshold=1.):
		mf_list=[]

		# mf_list is structured as follows:
		# - an outer list of V variables
		# - each item of the outer list contains C fuzzy set, one for each cluster

		number_of_variables = x_train.shape[1]
		for i in range(0, number_of_variables):
			xin=x_train[:,i]
			for j in range(0,partition_matrix.shape[1]):
				mfin = partition_matrix[:,j]
				mf, xx = self.convexMF(xin, mfin)
				prm = self.fitMF(xx, mf, mf_shape)
				mf_list.append(prm) 

		self._check_similarities(mf_list, number_of_variables, threshold=merge_threshold)

		#print(self._info_for_simplification)

		return mf_list

	def _check_graph(self, list_of_arcs):
		from networkx import DiGraph, Graph, is_strongly_connected
		G = DiGraph()
		nodelist = []
		for arc in list_of_arcs:
			G.add_edge(arc[0], arc[1])
			nodelist.append(arc[0])
			nodelist.append(arc[1])
		return self.is_subclique(G,list(set(nodelist)))

	def is_subclique(self,G,nodelist):
		H = G.subgraph(nodelist)
		n = len(nodelist)
		return H.size() == n*(n-1)/2

	def _extreme_values_for_variable(self, v):
		return min(self.xtrain.T[v]), max(self.xtrain.T[v])

	def _check_similarities(self, mf_list, number_of_variables,
			threshold=1.):

		number_of_clusters = len(mf_list)//number_of_variables
		
		from collections import defaultdict

		things_to_be_removed = defaultdict(list)

		for v in range(number_of_variables):
			#print (" * Checking similiarities for variable %d" % v)

			for c1 in range(number_of_clusters):
				for c2 in range(c1+1, number_of_clusters):
					#print ("Comparing fuzzy set of cluster %d with fuzzy set of cluster %d" % (c1,c2))

					index1 = v*number_of_clusters + c1
					index2 = v*number_of_clusters + c2
					funname1, params1 = mf_list[index1]
					funname2, params2 = mf_list[index2]

					if funname1== "gauss":
						
						from numpy import linspace, array

						mi, ma = self._extreme_values_for_variable(v)
						points = linspace(mi, ma, 100)

						first_cluster = array([self.gaussmf(x, params1[0], params1[1]) for x in points])
						second_cluster = array([self.gaussmf(x, params2[0], params2[1]) for x in points])

						intersection = sum([min(x,y) for x,y in zip(first_cluster, second_cluster)])
						union		= sum([max(x,y) for x,y in zip(first_cluster, second_cluster)])

						jaccardsim = (intersection/union)

						if jaccardsim>threshold:
							#print ("OMG it's the same!")
							things_to_be_removed[v].append([c1,c2,jaccardsim])

							#print("%.2f is fine" % jaccardsim)

					else:
						raise Exception("Not implemented yet")

		#print(things_to_be_removed)
		self._info_for_simplification = {}
		for k,value in things_to_be_removed.items():
			is_conn = self._check_graph(value)
			if is_conn:
				"""
				print (k, is_conn)
				print (value)
				print ( " I will retain %d" % value[0][0])
				"""
				retained = value[0][0]
				for element in value:
					if element[0]!=retained:
						self._info_for_simplification[(k,element[0])] = retained
					if element[1]!=retained:
						self._info_for_simplification[(k,element[1])] = retained

		print (" * %d antecedent clauses will be simplified using a threshold %.2f" % (len(self._info_for_simplification), threshold))

		self._info_for_simplification
		

	   
	def convexMF(self, xin, mfin, norm=1, nc=1000):
		
		# Calculates the convex membership function that envelopes a given set of
		# data points and their corresponding membership values. 
		
		# Input:
		# Xin: N x 1 input domain (column vector)
		# MFin: N x 1correspoding membership values 
		# nc: number of alpha cut values to consider (default=101)
		# norm: optional normalization flag (0: do not normalize, 1 : normalize, 
		# default=1)
		#
		# Output:
		# mf: membership values of convex function
		# x: output domain values	
		
		# Normalize the membership values (if requested)
		if norm == 1:
			mfin = np.divide(mfin, np.max(mfin))
		
		# Initialize auxilary variables
		acut = np.linspace(0,np.max(mfin),nc)
		mf= np.full(2*nc, np.nan)
		x=np.full(2*nc, np.nan)
		
		if np.any(mfin>0):
			x[0] = np.min(xin[mfin>0])
			x[nc]=np.max(xin[mfin>0])
			mf[0]=0
			mf[nc] = 0 
		
		# Determine the elements in the alpha cuts	
		for i in range(0,nc):
			if np.any(mfin>acut[i]):
				x[i]=np.min(xin[mfin>acut[i]])
				x[i+nc]=np.max(xin[mfin>acut[i]])
				mf[i]=acut[i]
				mf[i+nc]=acut[i]
				
		#Delete NaNs
		idx=np.isnan(x)
		x=x[idx==False]
		mf=mf[idx==False]  
		
		# Sort vectors based on membership value (descending order)
		indmf=mf.argsort(axis=0)
		indmf=np.flipud(indmf)
		mf=mf[indmf]
		x=x[indmf]
		
		# Find duplicate values for x and onlykeep the ones with the highest membership value
		_, ind = np.unique(x, return_index=True, return_inverse=False, return_counts=False, axis=None)
		mf=mf[ind]
		x=x[ind]
		
		# Sort vectors based on x value (ascending order)
		indx=x.argsort(axis=0)
		mf=mf[indx]
		x=x[indx]
		
		xval=np.linspace(np.min(x),np.max(x),nc)
		mf=np.interp(xval, x, mf, left=None, right=None, period=None)
		x=xval;
		return mf, x
	
	def fitMF(self,x,mf,mf_shape='gauss2'):
		# Fits parametrized membership functions to a set of pointwise defined 
		# membership values.
		#
		# Input:
		# x:  N x 1 domain of input variable
		# mf: N x 1 membership values for input data x 
		# shape: Type of membership function to fit (possible values: 'gauss', 
		# 'gauss2' and 'sigmf')
		#
		# Output:
		# param: matrix of membership function parameters
		
	
		if mf_shape == 'gauss':
			# Determine initial parameters
			mu = sum(x * mf) / sum(mf)
			sig = np.mean(np.sqrt(-((x-mu)**2)/(2*np.log(mf))))
			
			# Fit parameters to the data using least squares
#			print('mu=', mu, 'sig=', sig)
			param, _ = curve_fit(self.gaussmf, x, mf, p0 = [mu, sig], maxfev = 10000)
	   
		elif mf_shape == 'gauss2':
			# Determine initial parameters
			mu1 = x[mf>=0.95][0]
			mu2 = x[mf>=0.95][-1]
			xmf =x[mf>=0.5]
			sig1 = (mu1 - xmf[0])/(np.sqrt(2*np.log(2)))
			sig2 = (xmf[-1]-mu2)/(np.sqrt(2*np.log(2)))
			if sig1==0.0:
				sig1=0.1
			if sig2==0.0:
				sig2=0.1
			
			# Fit parameters to the data using least squares
#			print('mu1',mu1,'sig1',sig1,'mu2',mu2,'sig2',sig2)
			param, _ = curve_fit(self.gauss2mf, x, mf, p0 = [mu1, sig1, mu2, sig2], maxfev=1000)
			
		elif mf_shape == 'sigmf':
			# Determine initial parameters
			if np.argmax(mf)-np.argmin(mf) > 0:		 # if sloping to the right
				c = x[mf>=0.5][0]
				s = 1
			elif np.argmax(mf)-np.argmin(mf) < 0:	   # if sloping to the left
				c = x[mf<=0.5][0]
				s = -1
			# Fit parameters of the function to the data using non-linear least squares		   
			param, _ = curve_fit(self.sigmf, x, mf, p0 = [c, s])
		
		return mf_shape, param
		
	def gaussmf(self,x, mu, sigma, a=1):
		# x:  (1D array)
		# mu: Center of the bell curve (float)
		# sigma: Width of the bell curve (float)
		# a: normalizes the bell curve, for normal fuzzy set a=1 (float) 
		return a * np.exp(-(x - mu)**2 / (2 * sigma**2))
	
	
	def gauss2mf(self,x, mu1, sigma1, mu2, sigma2):
		# x: Data
		# mu1: Center of the leftside bell curve
		# sigma1: Standard deviation that determines the width of the leftside bell curve
		# mu2: Center of the rightside bell curve
		# sigma2: Standard deviation that determines the width of the rightside bell curve
		y = np.ones(len(x))
		idx1 = x <= mu1
		idx2 = x > mu2
		y[idx1] = self.gaussmf(x[idx1], mu1, sigma1)
		y[idx2] = self.gaussmf(x[idx2], mu2, sigma2)
		return y
	
	def sigmf(self,x, c, s):
		# x: data
		# b: x where mf is 0.5
		# c: Controls 'width' of the sigmoidal region about `b` (magnitude); also
		#	which side of the function is open (sign). A positive value of `a`
		#	means the left side approaches 0.0 while the right side approaches 1,
		#	a negative value of `c` means the opposite.
		return 1. / (1. + np.exp(- s * (x - c)))