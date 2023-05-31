from sklearn.metrics import log_loss, zero_one_loss
from scipy.special import expit
from scipy.spatial.distance import cdist
import numpy as np
import copy
import hashlib
import os
import pickle
class GA_Operations:
    def get_initial_pop(self, size, rng):
        raise NotImplementedError

    def crossover(self, parent1, parent2, rng):
        raise NotImplementedError

    def mutate(self, candidate, prob_mutation, mutation_scale, rng):
        raise NotImplementedError

    def get_utility_all(self, candidate_list):
        raise NotImplementedError

class Log_Regression(GA_Operations):
    def __init__(self, features, labels):
        assert len(features) == len(labels)
        self.features = features
        self.mod_features = np.append(features, np.ones((len(features),1)), axis=1)
        self.labels = labels
        self.cand_size = len(features[0])+1
        self.sensitivity = 1.0/len(self.labels)
        self.num_children = 2
        self.RANGE = 1
        self.initial_cost = 0

    def get_initial_pop(self, size, eps, rng):
        if size > 20:
            pop_1 = [np.zeros(self.cand_size) for _ in range(20)]
            pop_2 = [rng.uniform(low=-self.RANGE, high=self.RANGE, size=self.cand_size) for _ in range(size - 20)]
            return np.concatenate([pop_1, pop_2])
        else:
            return [rng.uniform(low=-self.RANGE, high=self.RANGE, size=self.cand_size) for _ in range(size)]

    def get_utility(self, candidate):
        alpha = candidate[:-1]
        beta = candidate[-1]
        predicitons = [expit(np.dot(x, alpha) + beta) for x in self.features]
        loss = self.zero_one(self.labels, predicitons)
        return loss

    def get_utility_all(self, candidate_list):
        candidate_list = np.array(candidate_list)
        predicitons = np.dot(self.mod_features, candidate_list.T)
        predicitons = expit(predicitons)
        loss = [self.zero_one(self.labels, pred) for pred in predicitons.T]
        return loss
    
    def mutate(self, candidate, prob_mutation, mutation_scale, rng):
        candidate = copy.deepcopy(candidate)
        for i in range(len(candidate)):
            if rng.uniform() <= prob_mutation:
                candidate[i] += rng.normal(scale=mutation_scale)
                candidate[i] = min(self.RANGE, max(candidate[i], -self.RANGE))
        return candidate
    
    def crossover(self, parent1, parent2, rng):
        indicator_vector = rng.integers(low=0, high=2, size=self.cand_size)
        child1 = np.where(indicator_vector, parent1, parent2)
        child2 = np.where(indicator_vector, parent2, parent1)
        return [child1, child2]

    def log(self, true_labels, pred_labels): 
        return -log_loss(true_labels, pred_labels, normalize=True)

    def zero_one(self, true_labels, pred_labels): 
        return -zero_one_loss(true_labels, np.rint(pred_labels).astype(int), normalize=True)

class K_Median(GA_Operations):
    def __init__(self, public_data, private_data, k, cache=True, distance_matrix=None):
        if distance_matrix is not None: #Used for random data
            self.distance_matrix = distance_matrix
            self.sensitivity = 10
        else:
            assert len(public_data[0]) == len(private_data[0])
            cache_id = hashlib.sha1(np.ascontiguousarray(public_data)).hexdigest()
            cache_id += hashlib.sha1(np.ascontiguousarray(private_data)).hexdigest()
            cache_id += hashlib.sha1(np.ascontiguousarray(np.array(k))).hexdigest()
            if os.path.exists('Cache/{}.p'.format(cache_id)):
                print("Loading distance matrix from cache")
                with open('Cache/{}.p'.format(cache_id), 'rb') as input_file:
                    self.distance_matrix = pickle.load(input_file)
            else:
                self.distance_matrix = cdist(public_data, private_data, metric='minkowski', p=1).T
                if cache:
                    with open('Cache/{}.p'.format(cache_id), 'wb') as out_file:
                        pickle.dump(self.distance_matrix, out_file)
            self.sensitivity = len(public_data[0])

        self.cand_size = k
        self.num_private = len(self.distance_matrix)
        self.num_public = len(self.distance_matrix[0])
        self.k = k
        self.num_children = 1
        self.initial_cost = 0
    
    def get_initial_pop(self, size, eps, rng):
        pop = [rng.choice(self.num_public, self.k, replace=False) for _ in range(size)]
        return pop

    def get_utility(self, candidate):
        distance_to_clusters = self.distance_matrix[:,candidate]
        utility = -np.sum(np.min(distance_to_clusters, axis=1))
        return utility
    
    def get_utility_all(self, candidate_list):
        assert len(candidate_list[0]) == self.k
        return [self.get_utility(cand) for cand in candidate_list]

    def mutate(self, candidate, prob_mutation, mutation_scale, rng):
        candidate = copy.deepcopy(candidate)
        for i in range(len(candidate)):
            if rng.uniform() <= prob_mutation:
                temp = rng.integers(low=0, high=self.num_public)
                while temp in candidate:
                    temp = rng.integers(low=0, high=self.num_public)
                candidate[i] = temp
        return candidate
    
    def crossover(self, parent1, parent2, rng):
        union =  list(set(parent1) | set(parent2))
        child = rng.choice(union, self.k, replace=False)
        return [child]
class K_Median_Clipped_new(K_Median):
    def __init__(self, public_data, private_data, k):
        assert len(public_data[0]) == len(private_data[0])
        cache_id = hashlib.sha1(np.ascontiguousarray(public_data)).hexdigest()
        cache_id += hashlib.sha1(np.ascontiguousarray(private_data)).hexdigest()
        cache_id += hashlib.sha1(np.ascontiguousarray(np.array(k))).hexdigest()
        cache_id += 'ClippedNew'
        if os.path.exists('Cache/{}.p'.format(cache_id)):
            print("Loading distance matrix from cache")
            with open('Cache/{}.p'.format(cache_id), 'rb') as input_file:
                self.distance_matrix = pickle.load(input_file)
        else:
            self.distance_matrix = cdist(public_data, private_data, metric='minkowski', p=1).T
            f = lambda x: np.maximum(np.minimum(len(public_data[0])/2 , x), len(public_data[0])/4)
            self.distance_matrix = f(self.distance_matrix)
            with open('Cache/{}.p'.format(cache_id), 'wb') as out_file:
                pickle.dump(self.distance_matrix, out_file)
        self.cand_size = k
        self.num_private = len(private_data)
        self.num_public = len(public_data)
        self.sensitivity = len(public_data[0])/4
        self.k = k
        self.num_children = 1
        self.initial_cost = 0
