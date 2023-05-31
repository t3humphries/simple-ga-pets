import numpy as np 
import pickle
from selection_functions import gumbel_select
from config import DEFAULT_GUPTA


''' Implemented by [Redacted for blind review] 2021 following Algorithm 2 of
    Anupam Gupta, Katrina Ligett, Frank McSherry, Aaron Roth, and Kunal Talwar. 2010. 
    Differentially private combinatorial optimization. In Proceedings of the twenty-first annual ACM-SIAM symposium on Discrete algorithms
    (SODA '10). Society for Industrial and Applied Mathematics, USA, 1106–1125.'''

def gupta_algo(params, util_obj):
    verbose = True
    rng = np.random.default_rng(seed=params['run'])
    k = util_obj.k
    num_public = util_obj.num_public
    senstitivity = util_obj.sensitivity
    num_iteration = round(6 * k * np.log(num_public)) #Theoretical fomula for num_iterations
    if params['iter_overide'] is not None:
        num_iteration = params['iter_overide']
    eps_per_selection = params['epsilon'] / (num_iteration + 1)
    #----------Testing with our privacy analysis----------------
    if params['adv_comp']:
        from composition_theorems import bounded_range_comp
        eps_per_selection = bounded_range_comp(num_iteration + 1, params['epsilon'], params['delta'])
    #----------------------------------------------------------

    #initialize pop
    init_solution = rng.choice(num_public, k, replace=False)

    best_solutions = []
    best_solutions_util = []
    current_solution = init_solution.copy()
    for iteration in range(num_iteration):
        possible_solutions = []
        for i, x in enumerate(current_solution):
            for y in range(num_public):
                if y not in current_solution:
                    new_solution = current_solution.copy()
                    new_solution[i] = y
                    possible_solutions.append(new_solution)
        utils = util_obj.get_utility_all(possible_solutions)
        selected_idx = gumbel_select(utils, 1, eps_per_selection, senstitivity, rng)[0]
        best_solution_this_round = possible_solutions[selected_idx]
        best_solutions.append(best_solution_this_round)
        best_solutions_util.append(utils[selected_idx])
        current_solution = best_solution_this_round
    
    final_chosen_idx = gumbel_select(best_solutions_util, 1, eps_per_selection, senstitivity, rng)[0]
    final_solution = best_solutions[final_chosen_idx]
    with open( 'Results/{}_{}_{}.p'.format(params['filename'], params['epsilon'], params['run']), "wb" ) as outfile:
            pickle.dump(final_solution, outfile)
    return None




