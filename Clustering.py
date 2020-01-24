import numpy as np
from scipy.spatial.distance import cdist

# To do:
# Give option to cluster in input or input-output space?
# More clustering methods

class Clusterer(object):
	def __init__(self, x_train, y_train, nr_clus):
		self.x_train=x_train
		self.y_train=y_train
		self.data=np.concatenate((self.x_train,self.y_train.reshape(len(self.y_train),1)),axis=1)
		self.nr_clus = nr_clus

	def cluster(self, method="fcm", **kwargs):
		try:
			m = kwargs["m"]
		except:
			m = 2

		if method == "fcm":
			try:
				max_iter = kwargs["max_iter"]
			except:
				max_iter = 1000
			try:
				error = kwargs["error"]
			except:
				error = 0.005
			try:
				seed = kwargs["seed"]
			except:
				seed = None

			centers, partition_matrix, jm = self._fcm(data=self.data, n_clusters=self.nr_clus, m=m, max_iter=max_iter, error=error)

		elif method == "fstpso":
			try:
				max_iter = kwargs["max_iter"]
			except:
				max_iter=100
			try:
				n_particles = kwargs["n_particles"]
			except:
				n_particles = None
			try:
				path_fit_dump = kwargs["path_fit_dump"]
			except:
				path_fit_dump = None
			try:
				path_sol_dump = kwargs["path_sol_dump"]
			except:
				path_sol_dump = None

			centers, partition_matrix, jm = self._fstpso(data=self.data, n_clusters=self.nr_clus, max_iter=max_iter, n_particles=n_particles, m=m, path_fit_dump=path_fit_dump, path_sol_dump=path_sol_dump)

		return centers, partition_matrix, jm

	def _fcm(self, data, n_clusters, m=2, max_iter=1000, error=0.005, seed=None):
		#data: 2d array, size (N, S). N is the number of instances; S is the number of variables
		#n_clusters: number of clusters
		#m: fuzzy clustering coefficient
		#max_it: maximum number of iterations, default=1000
		#error: stopping criterion, default=0.005
		#seed: seed for random initialization of u matrix
		
		n_instances = data.shape[0]
		
		#randomly initaliaze u
		if seed != None:
			np.random.seed(seed=seed)
		u = np.random.rand(n_instances, n_clusters)
		u = np.fmax(u, np.finfo(np.float64).eps)
		ut = u.T
		
		for it in range(0,max_iter):
			#copy old u matrix
			u_old = ut.copy()
			u_old /= np.ones((n_clusters, 1)).dot(np.atleast_2d(u_old.sum(axis=0)))
			u_old = np.fmax(u_old, np.finfo(np.float64).eps)
		
			#elevate to m
			um = u_old ** m
		
			#calculate cluster centers
			centers = um.dot(data) / (np.ones((data.shape[1], 1)).dot(np.atleast_2d(um.sum(axis=1))).T)
		
			#calculate distances
			dist = cdist(centers, data, metric='euclidean')
			dist = np.fmax(dist, np.finfo(np.float64).eps)
		
			#calculate objective
			jm = (um * dist ** 2).sum()
		
			#calculate new u matrix
			ut = dist ** (- 2. / (m - 1))
			ut /= np.ones((n_clusters, 1)).dot(np.atleast_2d(ut.sum(axis=0)))
		
			#stopping criterion
			if np.linalg.norm(ut - u_old) < error:
				break
		
		partition_matrix = ut.T
		return centers, partition_matrix, jm
	
	
	def _fstpso(self, data, n_clusters, max_iter=100, n_particles=None, m=2, path_fit_dump=None, path_sol_dump=None):
		#data: 2d array, size (N, S). N is the number of instances; S is the number of variables
		#n_clusters: number of clusters
		#max_iter: number of maximum iterations of FST-PSO, default is 100
		#n_particles: number of particles in the swarm, if None it is automatically set by FST-PSO
		#m: fuzzy clustering coefficient
		#path_fit_dump: path to the file where the best fitness score at each iteration will be dumped
		#path_sol_dump: path to the file where the best solution at each iteration will be dumped

		try:
			from fstpso import FuzzyPSO
		except:
			print("ERROR: please, pip install fst-pso to use this functionality.")

		n_instances = data.shape[0]
		n_variables = data.shape[1]

		#set search space boundaries
		bounds = [0]*n_variables
		for i in range(n_variables):
			x = min([row[i] for row in data])
			y = max([row[i] for row in data])
			bounds[i] = [x, y]

		search_space = []
		for i in bounds:
			search_space.extend([i]*n_clusters)

		#initializing FST-PSO
		FP = FuzzyPSO()
		FP.set_search_space(search_space)
		if n_particles != None: FP.set_swarm_size(n_particles)

		#generally better results are obtained with this rule disabled
		FP.disable_fuzzyrule_minvelocity()

		#fitness function definition
		def fitness(particle):
			particle = list(map(float,particle))
			centers = np.reshape(particle, (n_variables, n_clusters)).T

			#calculating fitness value of found solution
			dist = cdist(data, centers, metric='sqeuclidean')
			
			um = np.zeros(np.shape(dist))
			for i in range(np.shape(um)[0]):
				for j in range(np.shape(um)[1]):
					um[i][j] = np.sum(	np.power(	np.divide(	dist[i][j],dist[i])	,	float(1/(m-1))	)	)
			um = np.reciprocal(um)

			um_power = np.power(um,m)

			fitness_value = np.sum(np.multiply(um_power,dist))
			return fitness_value

		#fitness function setting
		FP.set_fitness(fitness, skip_test=True)

		#execute optimization
		result = FP.solve_with_fstpso(max_iter=max_iter, dump_best_fitness=path_fit_dump, dump_best_solution=path_sol_dump)

		#reshaping centers
		solution = list(map(float,result[0].X))
		centers = np.reshape(solution, (n_variables, n_clusters)).T
		
		#calculating membership matrix
		dist = cdist(data, centers, metric='sqeuclidean')
		um = np.zeros(np.shape(dist))
		for i in range(np.shape(um)[0]):
			for j in range(np.shape(um)[1]):
				um[i][j] = np.sum(	np.power(	np.divide(	dist[i][j],dist[i])	,	float(1/(m-1))	)	)
		partition_matrix = np.reciprocal(um)

		#final fitness value
		jm =  result[1]

		return centers, partition_matrix, jm

