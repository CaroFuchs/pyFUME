import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import norm

#from random import seed

# To do:
# Give option to cluster in input or input-output space?
# More clustering methods
# data input argument


class Clusterer(object):
    """
        Creates a new clusterer object that can cluster the (training) data in the input-output feature space. The user should specify the 'data'argument OR the 'x_train' and 'y_train' argument.
        
        Args:
            nr_clus: Number of clusters that should be identified in the data.  
            x_train: The input data (default = None).
            y_train: The output data (true label/golden standard) (default = None).
            data: The data to be clustered (default = None).
    """ 
    
    def __init__(self, nr_clus, x_train=None, y_train=None, data=None, relational_data=None, verbose=False):
        self.x_train=x_train
        self.y_train=y_train
        self.nr_clus = nr_clus
        self._verbose = verbose

        if data is None and (x_train is None or y_train is None) and relational_data is None:
            raise Exception("Please specify a valid dataset for clustering.")
        elif data is not None:
        	self.data=data
        elif relational_data is not None:
            self.relational_data=relational_data
        else:
            self.data=np.concatenate((self.x_train, np.expand_dims(self.y_train, axis=1)), axis=1)#.reshape(len(self.y_train),1)),axis=1)

                                     
    def cluster(self, method="fcm", **kwargs):
        """
        Clusters the data using the clustering method as specified by the user.
            
        Args:
                method: The method used for the clustering. The user can choose 'fcm' (fuzzy c-means), 'fst-pso' (fst-pso based clustering) and 'gk' (Gustafson-Kessel).
                **kwargs: Additional arguments to change settings of the clustering method.     
                
        Returns: 
                Tuple containing (centers, partion_matrix, jm)   
                    - centers: The location of the identified cluster centers.
                    -  partition_matrix: A matrix containing the cluster memberships of each data point to each of the clusters.
                    - jm: Fitness function of the best solution.
        """ 

        if self._verbose:
            print(" * Clustering method:", method)

        try:
            self.m = kwargs["m"]
        except:
            self.m = 2
            
        if method == "fcm" or method == 'gk' or method == 'GK' or method == 'Gustafson-Kessel' or method == 'gustafson-kessel' or method == 'g-k' or method == "pfcm" or method == "fst-pso" or method == "fstpso":
            assert self.data is not None or (self.x_train is not None and self.y_train is not None), 'Please specify the data tobe clustered.'
        elif method == 'RFCM' or method == "relational_clustering" or method == "relational" or method == "RC":
            assert self.relational_data is not None, 'Please specify the relational data matrix.'
            

        if method == "fcm":
            try:
                max_iter = kwargs["fcm_max_iter"]
            except:
                max_iter = 1000
            try:
                error = kwargs["fcm_error"]
            except:
                error = 0.005

            centers, partition_matrix, jm = self._fcm(data=self.data, n_clusters=self.nr_clus, m=self.m, max_iter=max_iter, error=error)
        
        elif method == 'gk' or method == 'GK' or method == 'Gustafson-Kessel' or method == 'gustafson-kessel' or method == 'g-k':
            try:
                max_iter = kwargs["gk_max_iter"]
            except:
                max_iter=1000
            try:
                error = kwargs["gk_error"]
            except:
                error = 0.005
            centers, partition_matrix, jm =self._gk(m=self.m, max_iter=max_iter) 

        elif method == "pfcm":
            try:
                n = kwargs["pfcm_n"]
            except:
                n = 2
            try:
                max_iter = kwargs["pfcm_max_iter"]
            except:
                max_iter = 1000
            try:
                error = kwargs["pfcm_error"]
            except:
                error = 0.005
            try:
                a = kwargs["pfcm_a"]
            except:
                a = 0.5
            try:
                b = kwargs["pfcm_b"]
            except:
                b = 0.5

            centers, partition_matrix, typicality_matrix, jm = self._pfcm(data=self.data, n_clusters=self.nr_clus, m=self.m, n=n, max_iter=max_iter, error=error, a=a, b=b)    

        elif method == "fst-pso" or method == "fstpso":
            try:
                max_iter = kwargs["fstpso_max_iter"]
            except:
                max_iter=100
            try:
                n_particles = kwargs["fstpso_n_particles"]
            except:
                n_particles = None
            try:
                path_fit_dump = kwargs["fstpso_path_fit_dump"]
            except:
                path_fit_dump = None
            try:
                path_sol_dump = kwargs["fstpso_path_sol_dump"]
            except:
                path_sol_dump = None
            centers, partition_matrix, jm = self._fstpso(data=self.data, n_clusters=self.nr_clus, max_iter=max_iter, n_particles=n_particles, m=self.m, path_fit_dump=path_fit_dump, path_sol_dump=path_sol_dump)
               
        elif method == 'RFCM' or method == 'rfcm' or method == "relational_clustering" or method == "relational" or method == "RC":
            try:
                max_iter = kwargs["RFCM_max_iter"]
            except:
                max_iter=100
            try:
                error = kwargs["RFCM_error"]
            except:
                error = 0.005
            try:
                initialization = kwargs["RFCM_initialization"]
            except:
                initialization = 'random_initialization'
            
            centers, partition_matrix, jm = self._rfcm(R=self.relational_data, c=self.nr_clus, m = self.m, epsilon = error, maxIter = max_iter, initType = initialization)            
        elif method == 'FKP' or method == 'fkp' or method == "fuzzy_k_protoypes":
            try:
                max_iter = kwargs["FKP_max_iter"]
            except:
                max_iter=100
            try:
                error = kwargs["FKP_error"]
            except:
                error = 0.005
            try:
                cat_ind = kwargs["categorical_indices"]
            except:
                import pandas as pd
                df = pd.DataFrame(self.data)
                catvar = df.apply(pd.Series.nunique) == 2
                if self._verbose: print('The following variables were recognized as being binaries, and are therefore treated as categorical variables for clustering: ', catvar[catvar ==True])
                cat_ind =  np.where(catvar)[0]
                # raise Exception('To utilize fuzzy K-prototypes clustering, the keyword "categorical_indices" should be specified.')
            
            centers, partition_matrix, jm = self._fuzzy_k_protoypes(data = self.data, categorical_indices = cat_ind, n_clusters = self.nr_clus, m = self.m, max_iter = max_iter, error=error)          
                    
        return centers, partition_matrix, jm

    def _fcm(self, data, n_clusters, m=2, max_iter=1000, error=0.005):
        #data: 2d array, size (N, S). N is the number of instances; S is the number of variables
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
    
    def _fuzzy_k_protoypes(self, data, categorical_indices, n_clusters, m=2, max_iter=1000, error=0.005):
        #data: 2d array, size (N, S). N is the number of instances; S is the number of variables
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
            prototypes = self._FKP_prototypes(um, data, n_clusters, categorical_indices)
        
            #calculate distances
            dist = self._FKP_distances(prototypes, data, categorical_indices, numerical_metric='euclidean')
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
        
        return prototypes, partition_matrix, jm
    
    def _FKP_prototypes(self, um, data, n_clusters, categorical_indices): 
        # Find the means to locate the cluster centers
        prototypes = um.dot(data) / (np.ones((data.shape[1], 1)).dot(np.atleast_2d(um.sum(axis=1))).T)
        
        # Transpose um
        um = um.T
        
        
        # Replace the mean with the mode if the variable is categorical
        for col_idx in categorical_indices:
            cats = np.unique([data[:,col_idx]])          # Find unique categories
            for clus in range(n_clusters):               # Find for each cluster the max summed membership
                freq=dict()
                for c in cats:
                    umpc = um[data[:,col_idx]==c,:]
                    freq[c]=sum(umpc[:,clus])   
                prototypes[clus, col_idx] = max(freq, key=freq.get)    
        return prototypes
    
    def _FKP_distances(self, prototypes, data, categorical_indices, numerical_metric='euclidean'):
        # If the data is categorical, determine if value is different from the mode
        # if data point == mode, distance = np.finfo(np.float64).eps
        # if data point ~= mode, distance i= 1
        dummy_proto = prototypes.copy()
        dummy_proto[:, categorical_indices] = 1
        
        distances = np.empty([prototypes.shape[0], data.shape[0]])
        for i in range(prototypes.shape[0]):
            dummy_data = data.copy()
            for cat in categorical_indices:
                dummy_data[:,cat] = np.where(data[:,cat]==prototypes[i,cat], 1,0)
            # If the data is numerical use euclidean distance
            distances[i,:] = cdist(dummy_proto[i,:, None].T, dummy_data, metric='euclidean')
        return distances
    
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

        #n_instances = data.shape[0]
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
                    um[i][j] = np.sum(    np.power(    np.divide(    dist[i][j],dist[i])    ,    float(1/(m-1))    )    )
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
                um[i][j] = np.sum(    np.power(    np.divide(    dist[i][j],dist[i])    ,    float(1/(m-1))    )    )
        partition_matrix = np.reciprocal(um)

        #final fitness value
        jm =  result[1]

        return centers, partition_matrix, jm
    
    def _pfcm(self, data, n_clusters, m=2, max_iter=1000, error=0.005, a=0.5, b=0.5, n=2):
        
        #data: Dataset to be clustered, with size M-by-N, where M is the number of data points
        #and N is the number of coordinates for each data point.
        #c : Number of clusters
        #m: fuzzy clustering coefficient (default = 2)
        #max_iter: Maximum number of iterations (default = 1000)
        #error : stopping criterion (default=0.005)
        #a: Relative importane of fuzzy membership (default = 0.5)
        #b: Relative importane of typicality values (default = 0.5) 
        #n: User-defined constant n (default = 2)

        #Return values:
        #centers: The locations of the found clusters centers
        #partition_matrix: Partition matrix
        #typicality_matrix: Typicality Matrix
        #jm: The objective funtion for U and T
        

        # Randomly initiaize the partitioning matrix and typicality matrix
        n_instances=len(data)
        partition_matrix = np.random.rand(n_instances, n_clusters)
        partition_matrix = np.fmax(partition_matrix, np.finfo(np.float64).eps)        # avoid 0's in the matrix
        typicality_matrix = np.random.rand(n_instances, n_clusters)
        typicality_matrix = np.fmax(typicality_matrix, np.finfo(np.float64).eps)        # avoid 0's in the matrix

        # Pre-allocation
        jm = np.zeros(shape=(max_iter, 1))
        g = np.zeros(shape=(n_clusters, data.shape[0]))


        for i in range(max_iter):
           
            # Perform one iteration of PFCM
            partition_matrix, typicality_matrix, centers, jm[i], g = self._pstepfcm(data=data, U=partition_matrix, T=typicality_matrix, m=m, a=a, b=b, n=n_clusters, g=g)
            
            # Stopping criterion: Stop if objective value does inrease < error
            if abs(jm[i] - jm[i-1]) < error:
                break

        return centers, partition_matrix, typicality_matrix, jm


    def _pstepfcm(self, data, U, T, g, m=2, n=2, a=0.5, b=0.5):
        #copy old u and t matrix and center locations
        um = U**m
        tf = T**n
        tfo = (1-T) ** n
        centers = (np.dot(a*um+b*tf, data).T/np.sum(a*um+b*tf, axis=1).T).T

        # Calculate distances between data points and cluster centers            
        dist = cdist(centers, data, metric='euclidean')
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        #calculate value of objective funtion
        jm = np.sum(np.sum(np.power(dist, 2)*(a*um+b*tf), axis=0)) + np.sum(g*np.sum(tfo, axis=0))

        #calculate new u and t matrix
        g = um*np.power(dist, 2)/(np.sum(um, axis=0))
        tmp = np.power(dist, (-2/(m-1)))
        U = tmp/(np.sum(tmp, axis=0))
        tmpt = np.power((b/g)*np.power(dist, 2), (1/(n-1)))
        T = 1/(1+tmpt)

        return U, T, centers, jm, g
    
    ### Gustafson-Kessel

    def _gk(self, m=2, max_iter=1000, error=0.01):
        
        # Initialize the partition matrix
        u = np.random.dirichlet(np.ones(self.data.shape[0]), size=self.nr_clus)
        
        centers = []
        iteration = 0
        
        while iteration < max_iter:
            u_old = u.copy()   # keep old partition matrix to evaluate stopping criterium
            
            # Caluculate the locations of the cluster centers
            centers = self._next_centers_gk(data=self.data, u=u, m=m)
            
            # Calculate the covariance matrix
            f = self._covariance_gk(data=self.data, v=centers, u=u, m=m)
            
            # Calculate the distance between cluster centers and data points
            dist = self._distance_gk(data=self.data, v=centers, f=f)
            
            # calculate objective
            jm = (u * dist ** 2).sum()
            
            # Update the partition matrix
            u = self._next_u_gk(dist)
            iteration += 1

            # Stopping criteria
            if norm(u - u_old) < error:
                iteration = max_iter
                
        u=np.transpose(u)        
        return centers, u, jm 

    def _next_centers_gk(self, data, u, m=2):
        um = u ** m
        return ((um @ data).T / um.sum(axis=1)).T

    def _covariance_gk(self, data, v, u, m=2):
        um = u ** self.m

        denominator = um.sum(axis=1).reshape(-1, 1, 1)
        temp = np.expand_dims(data.reshape(data.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        temp = np.matmul(temp, temp.transpose((0, 1, 3, 2)))
        numerator = um.transpose().reshape(um.shape[1], um.shape[0], 1, 1) * temp
        numerator = numerator.sum(0)

        return numerator / denominator

    def _distance_gk(self, data, v, f):
        dif = np.expand_dims(data.reshape(data.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        determ = np.sign(np.linalg.det(f)) * (np.abs(np.linalg.det(f))) ** (1 / self.m)
        det_time_inv = determ.reshape(-1, 1, 1) * np.linalg.pinv(f)
        temp = np.matmul(dif.transpose((0, 1, 3, 2)), det_time_inv)
        output = np.matmul(temp, dif).squeeze().T
        return np.fmax(output, 1e-8)

    def _next_u_gk(self, d):
        power = float(1 / (self.m - 1))
        d = d.transpose()
        new_u = d.reshape((d.shape[0], 1, -1)).repeat(d.shape[-1], axis=1)
        new_u = np.power(d[:, None, :] / new_u.transpose((0, 2, 1)), power)
        new_u = 1 / new_u.sum(1)
        new_u = new_u.transpose()
        return new_u
    
    ### Relational Clustering
    
    def _RF_init_centers(self, number_of_clusters, number_of_data_points, relational_data, method='random_initialization'):
        # Initialize the cluster cenetrs for  relational fuzzy clustering
        if method == 'random_initialization':       # Use random numbers
            V = np.random.rand(number_of_clusters,number_of_data_points)
            V = V/V.sum(axis=1, keepdims=True)
        elif method == 'randomly_choose_c_rows':    # Use randomly selected data points as initial centers
            idx = np.random.choice(number_of_data_points,  size=number_of_clusters, replace=False)
            V = relational_data[idx,:]
            V = V/V.sum(axis=1, keepdims=True)
       
        return V


    def _rfcm(self, R, c, m = 2, epsilon = 0.005, maxIter = 1000, initType = 'random_initialization'):
     
        #   Relational Fuzzy c-Means (RFCM) for clustering dissimilarity data as
        #	proposed in [1]. RFCM is the relational dual of Fuzzy c-Means (FCM), 
        #	so it expects the input relational matrix R to be Euclidean. 
        #
        # Output:
        #               U: fuzzy partition matrix / membership matrix
        #               V: cluster centers/coefficients
        #               jm: Fitness function of the best solution
        #
        # Input:
        # R         - the relational (dissimilarity) data matrix of size n x n
        # c         - number of clusters to be identified
        # m         - fuzzifier, default 2
        # epsilon   - convergence criteria, default 0.0001
        # initType  - initialize relational cluster centers V, default random initialization
        #               random initialization
        #               randomly choose c rows from D
        # maxIter   - the maximum number fo iterations, default 100
        #
        # Refs:
        #   [1] Hathaway, R. J., Davenport, J. W., & Bezdek, J. C. (1989). Relational duals 
        #   of the c-means clustering algorithms. Pattern recognition, 22(2), 205-212. 
               
        # Initialize variables
        D = np.array(R)         # Relational data
        n=len(D)                # Number of data points
        d = np.zeros([c,n])
        numIter=0
        stepSize=epsilon
        
        # Initialize the membership matrix randomly
        U = np.random.rand(c, n)
        U = np.fmax(U, np.finfo(np.float64).eps)
        
        # Initialize the (relational) cluster centers
        V = self._RF_init_centers(number_of_clusters = c, number_of_data_points = n, relational_data = D, method=initType);
        
        # Begin the main loop:
        while numIter<maxIter and stepSize >= epsilon:
            U0 = U;
            
            # Compute the relational distances d between clusters centers V and data points D
            d=np.zeros([c,n])
            for i in range(0,c):
                Vi=V[i,:]
                tmp1 = D@Vi.T
                tmp2 = Vi@D@Vi.T/2
                d[i,:]= tmp1 - tmp2
                print('d', np.min(d), np.max(d))    
            
            # Update the partition matrix U
            d = np.power(d,1/(m-1))
            tmp1 = np.divide(1,d)
            tmp2 = np.ones([c,1])*sum(tmp1)
            U = np.divide(np.divide(1,d),tmp2)
            
            # Update cluster centers V
            V = np.power(U,m)  
            V = V/V.sum(axis=1, keepdims=True)
            
            # Calculate the fitness value
            jm = (U * d ** 2).sum()
            
            # Update the step size
            stepSize=np.amax(abs(U-U0))
            print('U', np.min(U), np.max(U))
            print('U0', np.min(U0), np.max(U0))

            if self._verbose == True:
                print('pyFUME just finished iteration '+ str(numIter) + ' of the RFCM clustering algoritm.')
            
            numIter = numIter + 1;
        
        # Return the cluster center location, the membership matrix and the fitness value
        return V, U, jm
    
if __name__=='__main__':
    data = np.genfromtxt('data.csv', delimiter=',')
    cat_ind = [3]
    
    cl = Clusterer(data=data, nr_clus=3)

    # Perform FKP
    cluster_centers, partition_matrix, _ = cl._fuzzy_k_protoypes(data = data, categorical_indices = cat_ind, n_clusters = 2, m=2, max_iter=1000, error=0.005)

        

