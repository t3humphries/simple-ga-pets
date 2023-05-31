import numpy as np
import math
from pam_algo._k_medoids import *
from ga_util_objects import K_Median
from selection_functions import gumbel_select
from config import DEFAULT_JONES

''' Implemented by [Redacted for blind review] 2021 following Algorithm 2 of:
        Matthew Jones, Huy Nguyen, Thy Nguyen. 2021. 
        Differentially Private Clustering via Maximum Coverage 
        Proceedings of the AAAI Conference on Artificial Intelligence, 35(13), 11555-11563. 
        Retrieved from https://ojs.aaai.org/index.php/AAAI/article/view/17375 '''

def get_t_ball_at_point(t, center, distance_matrix):
    row = distance_matrix[:,center]
    elements_in_ball = np.argwhere(row <= t).flatten()
    return elements_in_ball
class Subset:
    def __init__(self, center_, elements_):
        self.center = center_
        self.elements = elements_
    
def max_coverage(R, subsets, epsilon, delta, m, rng):
    R = R.copy()
    eps_per_selection = epsilon / (2 * np.log(np.exp(1) / delta))
    chosen_subsets = []
    for i in range(m):
        remaining = set(R)
        if len(subsets) == 0:
            return chosen_subsets
        utilities = [len(s.elements.intersection(remaining)) for s in subsets]
        selected_set_idx = selected_set_idx = gumbel_select(utilities, 1,eps_per_selection, 1/2, rng)[0]
        # Note: this is not same exp mechanism, We use sensitivity = 0.5 so that we get eps * utility as the arg to exp func
        selected_set = subsets[selected_set_idx]
        chosen_subsets.append(selected_set)
        for x in selected_set.elements:
            if x in R:
                R.remove(x)
        subsets.remove(selected_set)
    return chosen_subsets

def jones_algo(params, data_objects):
    rng = np.random.default_rng(seed=params['run'])
    util_object = data_objects[0]
    public_data = data_objects[1]
    C = []
    r = math.ceil(1 + math.log(util_object.num_private, 1 + params['alpha']))
    V_prime = set(range(util_object.num_private))
    C = set([])
    for i in range(r):
        t_i = (1 + params['alpha']) ** i * util_object.sensitivity / util_object.num_private
        subsets = [Subset(i, set(get_t_ball_at_point(t_i, i, util_object.distance_matrix)).intersection(V_prime)) for i in range(util_object.num_public)]
        result = max_coverage(list(V_prime), subsets, params['epsilon']/2, params['delta'], math.ceil(2 * util_object.k * np.log(1/params['alpha'])), rng)
        V_i = []
        C_i = []
        for x in result:
            C_i.append(x.center)
            V_i.extend(x.elements)
        V_prime = V_prime - set(V_i)
        C = C | set(C_i)
    C = list(C)

    #Get the number of points at each cluster center in C
    distance_to_clusters = util_object.distance_matrix[:,C]
    cluster_assingment = np.argmin(distance_to_clusters, axis=1)
    unique, counts = np.unique(cluster_assingment, return_counts=True)
    final_counts = []
    for i in range(len(C)):
        if i in unique:
            idx = np.where(unique == i)
            final_counts.append(counts[idx][0])
        else:
            final_counts.append(0)

    #Add Laplace noise
    noisy_counts = [max(math.ceil(x + rng.laplace(scale=2/params['epsilon'])),0) for x in final_counts]
    #Create synthetic dataset
    synth_private_data = []
    synth_private_data_weights = []
    for i, x in enumerate(C):
        if noisy_counts[i] > 0:
            synth_private_data.append(public_data[x])
            synth_private_data_weights.append(noisy_counts[i])
    synth_private_data_weights = np.array(synth_private_data_weights)

    #Apply non-private K-medoid to this data
    new_util_object = K_Median(public_data, synth_private_data, util_object.k, cache=False)
    synth_distance_matrix = new_util_object.distance_matrix * synth_private_data_weights[:, np.newaxis]

    kmediods = K_Medoids_Custom(init='build', n_clusters=util_object.k).fit(synth_distance_matrix)
    solution = kmediods.medoid_indices_

    with open( 'Results/{}_{}_{}.p'.format(params['filename'], params['epsilon'], params['run']), "wb" ) as outfile:
        pickle.dump(solution, outfile)
    return None