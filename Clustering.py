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

    def fcm(self, data, n_clusters, m=2, max_iter=1000, error=0.005):
        #data: 2d array, size (N, S). N is the number of instances; S is the number of variables.
        #n_clusters: number of clusters
        #m: fuzzy clustering coefficient
        #max_it: maximum number of iterations, default=1000
        #error: stopping criterion, default=0.005
        #seed: seed for random initialization of u matrix
        
        n_instances = data.shape[0]
        
        #randomly initaliaze u
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
    
    

