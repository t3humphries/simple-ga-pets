import numpy as np
import pickle
import copy
from sklearn.metrics import log_loss, zero_one_loss
from scipy.special import expit
from selection_functions import gumbel_select, normal_selection
from config import DEFAULT_PRIV_GENE, DEFAULT_PRIV_EEM, DEFAULT_PRIV_LOCAL
''' Follows code published by Jun Zhang PrivLocal_Logistic.m
Source https://sourceforge.net/p/privgene/
Last Update: 2014-07-09

Initial Converstion to python by Lee and Kifer in: https://arxiv.org/pdf/1808.09501.pdf
Code: https://github.com/ppmlguy/DP-AGD

Modified by [Redacted for blind review] 2021 to work in my architecture'''

def priv_local(args, utility_obj):
    #Defaults
    verbose=False
    rng = np.random.default_rng(seed=args['run']) #For multithreading
    #print('starting priv local')

    max_performance = []
    avg_performace = []

    dataset_size, dimension_data = utility_obj.features.shape
    dimension_data += 1 #This is because privgene includes the label in the dimension
    generations = max(int(round((dataset_size * args['epsilon']) / 800.0)), 1) #Line 52 PrivLocal_Logistic.m
    if args['gen_overide'] is not None:
        generations = args['gen_overide']
    eps_per = args['epsilon'] / generations #Naive composition seen in line 72 of PrivLocal_Logistic.m
    #------------Testing with our privacy analysis----------------
    if args['adv_comp']:
        from composition_theorems import bounded_range_comp
        eps_per = bounded_range_comp(generations, args['epsilon'], args['delta'])
    #--------------------------------

    # mutation settings Line 25 of PrivLocal_Logistic.m
    mut_scale = 0.5 # = 5% of range [-5,5]
    mut_resize = 0.95

    # initialize \omega
    seed_remain = np.zeros((1, dimension_data)) #Line 55 PrivLocal_Logistic.m
    cur_scale = mut_scale #Line 58 PrivLocal_Logistic.m
    seed = np.zeros((2*dimension_data, dimension_data)) #Line 57 PrivLocal_Logistic.m

    for j in range(dimension_data): # Line 60 PrivLocal_Logistic.m
        mutation = np.zeros(dimension_data)
        mutation[j] = cur_scale
        seed[2*j, :] = seed_remain + mutation
        seed[2*j+1, :] = seed_remain - mutation

    for i in range(generations): #Line 66 PrivLocal_Logistic.m
        fittness = utility_obj.get_utility_all(seed) #Our version of line 68 PrivLocal_Logistic.m 
        sens = 2. * cur_scale / dataset_size # Line 70 PrivLocal_Logistic.m, normalized since we normalize our utility
        #------------Testing with our privacy analysis----------------
        if args['adv_comp']:
            sens = utility_obj.sensitivity
        #-----------------------------------------------------
        selected = gumbel_select(fittness, 1, eps_per, sens, rng)[0] #Our version of Line 72 PrivLocal_Logistic.m using gumbel noise
        seed_remain = seed[selected, :] #Line 73 PrivLocal_Logistic.m
        cur_scale *= mut_resize #Line 75 PrivLocal_Logistic.m

        for j in range(dimension_data): # Line 77 PrivLocal_Logistic.m
            mutation = np.zeros(dimension_data)
            mutation[j] = cur_scale
            seed[2*j, :] = seed_remain + mutation
            seed[2*j+1, :] = seed_remain - mutation

        #----------House Keeping -----------------------
        max_performance.append(seed[np.argmax(fittness), :]) #Added to track perfomance in each iteration
        avg_performace.append(np.std(fittness))
        if i % 10  == 0 and verbose:
            print('Generations {}, Avg Fit {}'.format(i, np.mean(fittness)))
    with open( 'Results/{}_{}_{}.p'.format(args['filename'], args['epsilon'], args['run']), "wb" ) as out_file:
        pickle.dump([max_performance, avg_performace, seed_remain], out_file )
    return None

''' Implemented by [Redacted for blind review] 2021 following Algorithm 1 PrivGene'''

def std_priv_gene(args, util_obj):
    #Defaults
    pop_size=200
    chosen_size=10
    mutuation_scale=0.5
    mutation_resize = 0.95
    verbose=False

    rng = np.random.default_rng(seed=args['run']) #For multithreading
    #util_obj = args['util_obj']
    dataset_size = len(util_obj.features)
    c = 0.00125 #From the experimental results in the paper
    generations = max(int(round((dataset_size * args['epsilon'] * c) / chosen_size)), 1) # using the formula with saftey to make sure at least one
    if args['gen_overide'] is not None:
        generations = args['gen_overide']
    
    eps_per_gen = args['epsilon'] / generations #Naive composition 
    eps_per_sel = eps_per_gen / chosen_size #Naive composition
    sensitivity = util_obj.sensitivity

    #Generating \omega^+, \omega^- and \omega as described in seciton 5.4
    positives = [np.concatenate([np.zeros(util_obj.cand_size -1), rng.uniform(low=0, high=util_obj.RANGE, size=1)]) for _ in range(10)] 
    negatives = [np.concatenate([np.zeros(util_obj.cand_size -1), rng.uniform(low=-util_obj.RANGE, high=0, size=1)]) for _ in range(10)] 
    randoms = [rng.uniform(low=-util_obj.RANGE, high=util_obj.RANGE, size=util_obj.cand_size) for _ in range(pop_size-20)]
    pop = np.concatenate([positives, negatives, randoms])

    pop_fit = util_obj.get_utility_all(pop)
    max_performance = []
    avg_performace = []
    if verbose:
        print('Initialized Population, Avg Fit {}'.format(np.mean(pop_fit)))
        print('Epsilon Overal {:.3f} Epsilon Per Choice {:.3f} Sensitivity {:.3f} Number of Selections {}'.format(args['epsilon'], 
        eps_per_sel, sensitivity, chosen_size * generations))
    for gen in range(generations - 1):#Priv gene saves 1 gen of privacy budget for the end
        #select top k parents
        
        if args['non_priv']:
            chosen_parents_indexs = normal_selection(pop_fit, chosen_size, -1, sensitivity, rng)
        else:
            chosen_parents_indexs = gumbel_select(pop_fit, chosen_size, eps_per_sel, sensitivity, rng)

        new_pop = []
        for _ in range(int(pop_size/2)):
            parent1_idx, parent2_idx = rng.choice(chosen_parents_indexs, size=2 , replace=False)
            parent1 = pop[parent1_idx]
            parent2 = pop[parent2_idx]
            child1, child2 = priv_gene_crossover(parent1, parent2, rng)
            child1 = priv_gene_mutate(child1, mutuation_scale, util_obj, rng)
            child2 = priv_gene_mutate(child2, mutuation_scale, util_obj, rng)
            new_pop.append(child1)
            new_pop.append(child2)
        
        pop = new_pop
        pop_fit = util_obj.get_utility_all(pop)
        mutuation_scale *= mutation_resize

        #----------House Keeping -----------------------
        max_performance.append(pop[np.argmax(pop_fit)])
        avg_performace.append(np.std(pop_fit))
        if gen % 10  == 0 and verbose:
                print('Generations {}, Avg Fit {}'.format(gen, np.mean(pop_fit)))
        #-----------------------------------------------
    
    if args['non_priv']:
        answer = normal_selection(pop_fit, 1, -1, sensitivity, rng)[0]
    else:
        answer = gumbel_select(pop_fit, 1, eps_per_gen, sensitivity, rng)[0]
    with open( 'Results/{}_{}_{}.p'.format(args['filename'], args['epsilon'], args['run']), "wb" ) as outfile:
        pickle.dump([max_performance, avg_performace, pop[answer]], outfile)
    return None 

def priv_eem(args, util_obj):
    #Defaults
    pop_size=200
    mutuation_scale=0.5
    mutation_resize = 0.95
    verbose=False

    rng = np.random.default_rng(seed=args['run']) #For multithreading
    dataset_size = len(util_obj.features)
    c = 0.00125 #From the experimental results in the paper
    generations = max(int(round((dataset_size * args['epsilon'] * c))), 1) # using the formula with saftey to make sure at least one
    if args['gen_overide'] is not None:
        generations = args['gen_overide']
    
    eps_per_gen = args['epsilon'] / generations #Naive composition 
    sensitivity = 2. * mutuation_scale / dataset_size if not args['adv_comp'] else util_obj.sensitivity

    #------------Testing with our privacy analysis----------------
    if args['adv_comp']:
        from composition_theorems import bounded_range_comp
        eps_per_gen = bounded_range_comp(generations, args['epsilon'], args['delta'])
    #--------------------------------
    #Generating \omega^+, \omega^- and \omega as described in seciton 5.4
  
    positives = [np.concatenate([np.zeros(util_obj.cand_size -1), rng.uniform(low=0, high=util_obj.RANGE, size=1)]) for _ in range(10)] 
    negatives = [np.concatenate([np.zeros(util_obj.cand_size -1), rng.uniform(low=-util_obj.RANGE, high=0, size=1)]) for _ in range(10)] 
    randoms = [rng.uniform(low=-util_obj.RANGE, high=util_obj.RANGE, size=util_obj.cand_size) for _ in range(pop_size-20)]
    pop = np.concatenate([positives, negatives, randoms])

    pop_fit = util_obj.get_utility_all(pop)
    max_performance = []
    avg_performace = []
    if verbose:
        print('Initialized Population, Avg Fit {}'.format(np.mean(pop_fit)))
        print('Epsilon Overal {:.3f} Epsilon Per Choice {:.3f} Sensitivity {:.3f} Number of Selections {}'.format(args['epsilon'], 
        sensitivity, generations))
    for gen in range(generations - 1):#Priv gene saves 1 gen of privacy budget for the end
        #select top k parents
        chosen_parents_indexs = gumbel_select(pop_fit, 1, eps_per_gen, sensitivity, rng)
        new_pop = [priv_gene_mutate(pop[chosen_parents_indexs[0]], mutuation_scale, util_obj, rng) for _ in range(pop_size)]
        pop = new_pop
        pop_fit = util_obj.get_utility_all(pop)
        mutuation_scale *= mutation_resize
        if not args['adv_comp']:
            sensitivity = 2. * mutuation_scale / dataset_size

        #----------House Keeping -----------------------
        max_performance.append(pop[np.argmax(pop_fit)])
        avg_performace.append(np.std(pop_fit))
        if gen % 10  == 0 and verbose:
                print('Generations {}, Avg Fit {}'.format(gen, np.mean(pop_fit)))
        #-----------------------------------------------

    answer = gumbel_select(pop_fit, 1, eps_per_gen, sensitivity, rng)[0]
    with open( 'Results/{}_{}_{}.p'.format(args['filename'], args['epsilon'], args['run']), "wb" ) as outfile:
        pickle.dump([max_performance, avg_performace, pop[answer]], outfile)
    return None 
#----------------Priv Gene Helper Functions------------------------------

def priv_gene_mutate(candidate, mut_scale, util_obj, rng):
    candidate = copy.deepcopy(candidate)
    selected_candidate = rng.choice(range(len(candidate)))
    candidate[selected_candidate] += rng.uniform(low=-1.0, high=1.0) * mut_scale
    #truncate if out of range
    candidate[selected_candidate] = min(util_obj.RANGE, max(candidate[selected_candidate], -util_obj.RANGE)) 
    return candidate

def priv_gene_crossover(parent1, parent2, rng):
    crossover_point = rng.integers(low=1, high=len(parent1)-1)
    child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
    return child1, child2
class Priv_Gene_Utility:
    def __init__(self, features, labels, type='log'):
        assert len(features) == len(labels)
        self.features = features
        self.mod_features = np.append(features, np.ones((len(features),1)), axis=1)
        self.labels = labels
        self.cand_size = len(features[0])+1
        self.type = type 
        self.sensitivity = 0
        self.RANGE = 5 
        if self.type == 'log':
            #Note this is overridden when using eem
            self.sensitivity = (len(self.features[0]) * self.RANGE + 1) /len(self.labels) 
        elif self.type == 'zero_one':
            self.sensitivity = 1.0/len(self.labels)
        else:
            raise RuntimeError("Invalid Loss Type")

    def get_utility(self, candidate):
        alpha = candidate[:-1]
        beta = candidate[-1]
        predicitons = [expit(np.dot(x, alpha) + beta) for x in self.features]
        if self.type == 'log':
            loss = self.log(self.labels, predicitons) 
        elif self.type == 'zero_one':
            loss = self.zero_one(self.labels, predicitons)
        return loss

    def get_utility_all(self, candidate_list):
        candidate_list = np.array(candidate_list)
        predicitons = np.dot(self.mod_features, candidate_list.T)
        predicitons = expit(predicitons)
        if self.type == 'log':
            loss = [self.log(self.labels, pred) for pred in predicitons.T]
        elif self.type == 'zero_one':
            loss = [self.zero_one(self.labels, pred) for pred in predicitons.T]
        return loss

    def log(self, true_labels, pred_labels): 
        return -log_loss(true_labels, pred_labels, normalize=True)

    def zero_one(self, true_labels, pred_labels): 
        return -zero_one_loss(true_labels, np.rint(pred_labels).astype(int), normalize=True)