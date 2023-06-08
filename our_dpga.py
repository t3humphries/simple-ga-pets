import numpy as np
import copy
import pickle
import os
import argparse
from ga_util_objects import Log_Regression, K_Median_Clipped_new
from config import DEFAULT_PARMS
from composition_theorems import bounded_range_comp
from selection_functions import gumbel_select

def saving_GA(params, ga_ops_dict, verbose=False):
    if os.path.exists('Results/{}_{}_{}.p'.format(params['filename'], params['epsilon'], params['run'])):
        print('File Exists Not repeating')
        return None
    ga_ops = ga_ops_dict[params['ga_ops']]
    rng = np.random.default_rng(seed=params['run'])
    total_choices = params['max_gen'] * params['chosen_size'] + ga_ops.initial_cost + 1 #extra one for the final selection

    noise_param = params['comp'](total_choices, params['epsilon'], params['delta'])

    pop = ga_ops.get_initial_pop(params['pop_size'], noise_param, rng)
    pop_fit = ga_ops.get_utility_all(pop)

    max_performance = []
    std_dev = []
    if verbose:
        print('Initialized Population, Avg Fit {}'.format(np.mean(pop_fit)))
        print('Epsilon Overal {:.3f} Epsilon Per Choice {:.3f} Sensitivity {:.3f} Number of Selections {}'.format(params['epsilon'], noise_param, ga_ops.sensitivity, total_choices))
    new_pop_size = params['pop_size'] - params['chosen_size'] if params['elitism'] else params['pop_size']

    for gen in range(params['max_gen']):
        #select top k parents
        chosen_parents_indexs = params['select'](pop_fit, params['chosen_size'], noise_param, ga_ops.sensitivity, rng)
        new_pop = [copy.deepcopy(pop[x]) for x in chosen_parents_indexs] if params['elitism'] else []
        for _ in range(int(new_pop_size/ga_ops.num_children)):
            parents = [ pop[rng.choice(chosen_parents_indexs)] for _ in range(2)]

            if rng.uniform() <= params['p_cross']:
                children = ga_ops.crossover(parents[0], parents[1], rng)
            else:
                children = [copy.deepcopy(parents[i]) for i in range(ga_ops.num_children)]
            
            children = [ga_ops.mutate(children[i], params['p_mutation'], params['mutation_scale'], rng) for i in range(ga_ops.num_children)]
            new_pop.extend(children)
        
        pop = new_pop
        pop_fit = ga_ops.get_utility_all(pop)

        max_performance.append(pop[np.argmax(pop_fit)])
        std_dev.append(np.std(pop_fit))

        if gen % 10  == 0 and verbose:
            print('Generations {}, Avg Fit {}'.format(gen, np.mean(pop_fit)))

    answer = params['select'](pop_fit, 1, noise_param, ga_ops.sensitivity, rng)[0]
    if not os.path.exists('Results/'):
        os.makedirs('Results/')
    with open( 'Results/{}_{}_{}.p'.format(params['filename'], params['epsilon'], params['run']), "wb" ) as outfile:
        pickle.dump([max_performance, std_dev, pop[answer]], outfile)
    return None


if __name__ == "__main__":
    def read_data_get_utility_obj(dataset, problem):
        '''Helper function to generate the GA objects'''
        data_dict = pickle.load(open('processed_datasets/' + dataset + '.p', 'rb'))
        if problem == 'LR':
            ga_object = Log_Regression(data_dict['test_train_sets']['train_x'],
                                            data_dict['test_train_sets']['train_y'])
                                    
            dataset_size = len(data_dict['test_train_sets']['train_x'])
        elif problem.startswith('KM'):
            k = int(problem.split('_')[1])
            ga_object = K_Median_Clipped_new(data_dict['clustering_sets']['public'],
                                    data_dict['clustering_sets']['private'], k)
            dataset_size = len(data_dict['clustering_sets']['private'])
        else:
            raise RuntimeError('Invalid Problem')
        delta = 1/(dataset_size**1.1)
        return ga_object, delta
    
    # Read the arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Spam',
                        help='Name of the dataset, options [Adult, Credit, Mushrooms, Spam]')
    parser.add_argument('--problem_name', type=str, default='KM_16',
                        help='Which objective to optimise out of [LR, KM_4, KM_16]')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed to use')
    parser.add_argument('--pop_size', type=int, default=200,
                        help='Size of the GA Population')
    parser.add_argument('--chosen_size', type=int, default=10,
                        help='Number of parents to select in each generation')
    parser.add_argument('--elitism', type=bool, default=True,
                        help='Should we keep the selected parents each generation')
    parser.add_argument('--p_cross', type=float, default=0.5,
                        help='Probability to do crossover at all')
    parser.add_argument('--p_mutation', type=float, default=0.01,
                        help='Probability of mutating each genome')
    parser.add_argument('--mutation_scale', type=float, default=0.1,
                        help='Standard devation of the Gaussian used for muation')
    parser.add_argument('--max_gen', type=int, default=100,
                        help='Maximum number of generations')
    parser.add_argument('--epsilon', type=float, default=1.0,
                        help='Total Epsilon value for privacy budget')
    parser.add_argument('--delta', type=float, default=None,
                        help='Total Delta value for privacy budget (default is 1/n^1.1)')
    parser.add_argument('--filename', type=str, default='Basic_GA',
                        help='Where to save the output')
    args = parser.parse_args()
    print(vars(args))
    
    #Parse into the parameters and GA_dict
    ga_dict, delta = read_data_get_utility_obj(args.dataset_name, args.problem_name)
    GA_ARGS = {
        'ga_ops': 'None',
        'filename': args.filename,
        'run': args.seed,
        'pop_size': args.pop_size,
        'chosen_size': args.chosen_size,
        'elitism': args.elitism,
        'p_cross': args.p_cross,
        'p_mutation': args.p_mutation,
        'mutation_scale': args.mutation_scale,
        'max_gen': args.max_gen,
        'epsilon': args.epsilon,
        'delta': args.delta,
        'select': gumbel_select,
        'comp': bounded_range_comp 
    }
    if args.delta is None:
        GA_ARGS['delta'] = delta
    #Run and save to pickle
    saving_GA(GA_ARGS, {'None':ga_dict}, verbose=True)