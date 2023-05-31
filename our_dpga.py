import numpy as np
import copy
import pickle
import os

def saving_GA(params, ga_ops_dict, verbose=False):
    if os.path.exists('Results/{}_{}_{}.p'.format(params['filename'], params['epsilon'], params['run'])):
        #print('File Exists Not repeating')
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
    with open( 'Results/{}_{}_{}.p'.format(params['filename'], params['epsilon'], params['run']), "wb" ) as outfile:
        pickle.dump([max_performance, std_dev, pop[answer]], outfile)
    return None