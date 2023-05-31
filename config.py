from composition_theorems import tight_adv_comp, bounded_range_comp, naive_comp, none_private
from selection_functions import gumbel_select, exp_noisy_max_select, noisy_max_select, normal_selection

DEFAULT_PARMS = {
    'ga_ops' : None,
    'filename' : 'Basic_GA',
    'run' : 0,
    'pop_size' : 200,
    'chosen_size' : 10,
    'elitism' : True,
    'p_cross' : 0.5,
    'p_mutation' : 0.01,
    'mutation_scale' : 0.1,
    'max_gen' : 100,
    'epsilon' : 10.0,
    'delta' : 1e-5,
    'select' : gumbel_select,
    'comp' : bounded_range_comp
}

DEFAULT_GUPTA= {
        'filename' : 'Test',
        'run' : 0,
        'util_obj' : None,
        'epsilon' : 10.0,
        'iter_overide' : None,
        'adv_comp': False,
        'delta' : 1e-5
    }

DEFAULT_JONES= {
        'filename' : 'Test',
        'run' : 0,
        'public_data' : None,
        'util_obj' : None,
        'epsilon' : 10.0,
        'delta' : 1e-5,
        'alpha' : 0.1
    }

DEFAULT_PRIV_GENE= {
        'filename' : 'Test',
        'run' : 0,
        'util_obj' : None,
        'epsilon' : 10.0,
        'gen_overide' : None,
        'non_priv' : False
    }

DEFAULT_PRIV_LOCAL= {
    'filename' : 'Test',
    'run' : 0,
    'util_obj' : None,
    'epsilon' : 10.0,
    'gen_overide' : None,
    'delta' : 1e-5,
    'adv_comp': False
    }

DEFAULT_PRIV_EEM= {
    'filename' : 'Test',
    'run' : 0,
    'util_obj' : None,
    'epsilon' : 10.0,
    'gen_overide' : None,
    'delta' : 1e-5,
    'adv_comp': False
    }
SELECTION_ALGS = { 'gumbel' : gumbel_select, 'permflip': exp_noisy_max_select, 'noisym' : noisy_max_select, 'non-priv' : normal_selection}
COMP_ALGS = {'kairouz' : tight_adv_comp, 'br': bounded_range_comp, 'naive': naive_comp, 'non-priv' : none_private}
GENS_EPS = [0.01, 0.0316227766017, 0.1, 0.316227766017, 1, 3.16227766017, 10, 31.6227766017]
DATASETS=['Adult', 'Credit', 'Spam', 'Mushrooms']