import argparse
import os
import copy
import pickle
import tqdm
import ray
from config import *
from ga_util_objects import Log_Regression, K_Median, K_Median_Clipped_new
from our_dpga import saving_GA
from privgene import std_priv_gene, priv_eem, priv_local, Priv_Gene_Utility
from gupta import gupta_algo
from jones import jones_algo

def read_data_get_utility_obj(dataset, problem, vanilla_loss=False):
    data_dict = pickle.load(open('processed_datasets/' + dataset + '.p', 'rb'))
    if problem == 'LR':
        if vanilla_loss:
            ga_object = Priv_Gene_Utility(data_dict['test_train_sets']['train_x'],
                                        data_dict['test_train_sets']['train_y'], type='log')
        else:
            ga_object = Log_Regression(data_dict['test_train_sets']['train_x'],
                                        data_dict['test_train_sets']['train_y'])
                                
        dataset_size = len(data_dict['test_train_sets']['train_x'])
    elif problem.startswith('KM'):
        k = int(problem.split('_')[1])
        if vanilla_loss:
            ga_object = K_Median(data_dict['clustering_sets']['public'],
                                data_dict['clustering_sets']['private'], k)
        else:
            ga_object = K_Median_Clipped_new(data_dict['clustering_sets']['public'],
                                data_dict['clustering_sets']['private'], k)
        dataset_size = len(data_dict['clustering_sets']['private'])
    else:
        raise RuntimeError('Invalid Problem')
    delta = 1/(dataset_size**1.1)
    return ga_object, delta

def selection_experiment(dataset='Adult', problem='LR', runs=100):
    jobs = [] 
    print('Preping Selection Experiment {} {}'.format(dataset, problem))
    directory = 'Selection_Experiment/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    loss_type = False if problem == 'LR' else True # Use zero one loss for the selection experiment
    ga_object, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=loss_type)
    ga_dict = {'std' : ga_object}
    mutation_rate = 1 / (ga_object.cand_size + 1)
    for select, comp in [('gumbel', 'kairouz'), ('gumbel', 'br'), ('permflip','kairouz'), ('noisym','kairouz')]:
        params = copy.deepcopy(DEFAULT_PARMS)
        params['p_mutation'] = mutation_rate
        params['delta'] = delta
        params['ga_ops'] = 'std'
        params['select'] = SELECTION_ALGS[select]
        params['comp'] = COMP_ALGS[comp]
        params['filename'] = directory + 'ga_{}_{}'.format(select, comp)
        for eps in GENS_EPS:
            for run in range(runs):
                job = params.copy()
                job['epsilon'] = eps 
                job['run'] = run
                jobs.append(job)
    return jobs, ga_dict

def utility_experiment(dataset='Adult', problem='KM', runs=100):
    jobs = [] 
    print('Preping Utility Experiment {} {}'.format(dataset, problem))
    directory = 'Utility_Experiment/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    ga_object, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=True)
    ga_dict = {'std' : ga_object}
    mutation_rate = 1 / (ga_dict['std'].cand_size + 1)

    params = copy.deepcopy(DEFAULT_PARMS)
    params['p_mutation'] = mutation_rate
    params['delta'] = delta
    params['ga_ops'] = 'std'
    params['filename'] = directory + 'ga_std'
    for eps in GENS_EPS:
        for run in range(runs):
            job = params.copy()
            job['epsilon'] = eps 
            job['run'] = run
            jobs.append(job)

    ga_dict['clipped'], delta = read_data_get_utility_obj(dataset, problem)
    params = copy.deepcopy(DEFAULT_PARMS)
    params['p_mutation'] = mutation_rate
    params['delta'] = delta
    params['ga_ops'] = 'clipped'
    params['filename'] = directory + 'ga_clip'
    for eps in GENS_EPS:
        for run in range(runs):
            job = params.copy()
            job['epsilon'] = eps 
            job['run'] = run
            jobs.append(job)
    return jobs, ga_dict

def generations_experiment(dataset='Adult', problem='LR', runs=30):
    jobs = [] 
    print('Preping Generations Experiment {} {}'.format(dataset, problem))
    directory = 'Generations_Experiment/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    loss_type = True if problem.startswith('KM') else False
    ga_object, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=loss_type)
    ga_dict={'std' : ga_object}
    mutation_rate = 1 / (ga_object.cand_size + 1)
    for generations in [0, 1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 75, 80, 90, 100, 120, 140, 160]:
        params = copy.deepcopy(DEFAULT_PARMS)
        params['ga_ops'] = 'std'
        params['filename'] = directory + 'ga_{}'.format(generations)
        params['p_mutation'] = mutation_rate
        params['delta'] = delta
        for eps in GENS_EPS:
            for run in range(runs):
                job = params.copy()
                job['max_gen'] = generations
                job['epsilon'] = eps 
                job['run'] = run
                jobs.append(job)
    return jobs, ga_dict

def PrivGene_generations_experiment(dataset='Adult', problem='LR', runs=30):
    jobs = [] 
    print('Preping Generations Experiment {} {}'.format(dataset, problem))
    directory = 'Generations_Experiment_PG/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    ga_object, _ = read_data_get_utility_obj(dataset, problem, vanilla_loss=True)
    for generations in [1, 2, 4, 6, 8, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160]:
        params = copy.deepcopy(DEFAULT_PRIV_GENE)
        params['ga_ops'] = 'std'
        params['filename'] = directory + 'PrivGene_{}'.format(generations)
        for eps in GENS_EPS:
            for run in range(runs):
                job = params.copy()
                job['gen_overide'] = generations
                job['epsilon'] = eps 
                job['run'] = run
                jobs.append(job)
    return jobs, ga_object

def grid_experiment(dataset='Adult', problem='LR', runs=30):
    jobs = [] 
    print('Preping Grid Search Experiment {} {}'.format(dataset, problem))
    directory = 'Grid_Experiment/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    loss_type = True if problem.startswith('KM') else False
    ga_object, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=loss_type)
    ga_objects={'std' : ga_object}
    cand_size = ga_objects['std'].cand_size + 1
    for generations in [2, 10, 20, 50, 75, 100, 120]:
        for select in [5, 10 , 20]:
            for mut in [2/cand_size, 1.5/cand_size, 1/cand_size, 0.75/cand_size]:
                for cross in [0.35, 0.5, 0.65]:
                    params = copy.deepcopy(DEFAULT_PARMS)
                    params['ga_ops'] = 'std'
                    if problem=='LR':
                        params['filename'] = directory + 'ga_{}_{}_{:.4f}_{}'.format(generations, select, mut, cross)
                    else:
                        params['filename'] = directory + 'ga_{}_{}_{:.2f}_{}'.format(generations, select, mut, cross)
                    params['delta'] = delta
                    for eps in GENS_EPS:
                        for run in range(runs):
                            job = params.copy()
                            job['max_gen'] = generations
                            job['chosen_size'] = select
                            job['p_cross'] = cross
                            job['p_mutation'] = mut
                            job['epsilon'] = eps 
                            job['run'] = run
                            if not os.path.exists('Results/{}_{}_{}.p'.format(job['filename'], job['epsilon'], job['run'])):
                                jobs.append(job)
    return jobs, ga_objects

def non_priv_grid_experiment(dataset='Adult', problem='LR', runs=30):
    jobs = [] 
    print('Preping Non-Private_Grid Search Experiment {} {}'.format(dataset, problem))
    directory = 'Non_Priv_Grid_Experiment/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    loss_type = True if problem.startswith('KM') else False
    ga_object, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=loss_type)
    ga_objects={'std' : ga_object}
    cand_size = ga_objects['std'].cand_size + 1
    for generations in [10, 20, 50, 100, 150, 200]:
        for select in [5, 10 , 20, 40]:
            for mut in [2/cand_size, 1.5/cand_size, 1/cand_size, 0.75/cand_size]:
                for cross in [0.35, 0.5, 0.65, 0.8]:
                    params = copy.deepcopy(DEFAULT_PARMS)
                    params['ga_ops'] = 'std'
                    if problem=='LR':
                        params['filename'] = directory + 'ga_{}_{}_{:.4f}_{}'.format(generations, select, mut, cross)
                    else:
                        params['filename'] = directory + 'ga_{}_{}_{:.2f}_{}'.format(generations, select, mut, cross)
                    for run in range(runs):
                        job = params.copy()
                        job['select'] = SELECTION_ALGS['non-priv']
                        job['comp'] = COMP_ALGS['non-priv']
                        job['max_gen'] = generations
                        job['chosen_size'] = select
                        job['p_cross'] = cross
                        job['p_mutation'] = mut
                        job['run'] = run
                        if not os.path.exists('Results/{}_{}_{}.p'.format(job['filename'], job['epsilon'], job['run'])):
                            jobs.append(job)
    return jobs, ga_objects

def grid_rerun(dataset='Adult', problem='LR', runs=100):
    jobs = [] 
    print('Preping Grid Re Runs {} {}'.format(dataset, problem))
    directory = 'Grid_Final/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    loss_type = True if problem.startswith('KM') else False
    ga_object, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=loss_type)
    ga_objects={'std' : ga_object}

    grid_params = pickle.load(open('misc_p_files/grid_params.p', 'rb'))[dataset+problem]

    params = copy.deepcopy(DEFAULT_PARMS)
    params['ga_ops'] = 'std'
    params['filename'] = directory + 'ga_opt'
    params['delta'] = delta
    for i, eps in enumerate(GENS_EPS):
        generations, select, mut, cross = grid_params[i][:4]
        for run in range(runs):
            job = params.copy()
            job['max_gen'] = int(generations)
            job['chosen_size'] = int(select)
            job['p_cross'] = float(cross)
            job['p_mutation'] = float(mut)
            job['epsilon'] = eps 
            job['run'] = run
            if not os.path.exists('Results/{}_{}_{}.p'.format(job['filename'], job['epsilon'], job['run'])):
                jobs.append(job)
    # Non-Private Optimal Runs
    np_grid_params = pickle.load(open('misc_p_files/np_grid_params.p', 'rb'))[dataset+problem]
    generations, select, mut, cross = np_grid_params[:4]
    for run in range(runs):
        job = params.copy()
        job['max_gen'] = int(generations)
        job['chosen_size'] = int(select)
        job['p_cross'] = float(cross)
        job['p_mutation'] = float(mut)
        job['select'] = SELECTION_ALGS['non-priv']
        job['comp'] = COMP_ALGS['non-priv']
        job['epsilon'] = -1
        job['run'] = run
        jobs.append(job)
    gens_mapping = { '0.01': '10',
                '0.0316227766017': '10',
                '0.1': '20',
                '0.316227766017' : '50',
                '1' : '75',
                '3.16227766017' : '100',
                '10' : '120',
                '31.6227766017' : '120'}
    mutation_rate = 1 / (ga_objects['std'].cand_size + 1)
    params = copy.deepcopy(DEFAULT_PARMS)
    params['ga_ops'] = 'std'
    params['filename'] = directory + 'ga_std'
    params['delta'] = delta
    params['p_mutation'] = mutation_rate
    for eps in GENS_EPS:
        for run in range(runs):
            job = params.copy()
            job['max_gen'] = int(gens_mapping[str(eps)])
            job['epsilon'] = eps 
            job['run'] = run
            if not os.path.exists('Results/{}_{}_{}.p'.format(job['filename'], job['epsilon'], job['run'])):
                jobs.append(job)
    #Non-private default run
    for run in range(runs):
        job = params.copy()
        job['max_gen'] = 150
        job['select'] = SELECTION_ALGS['non-priv']
        job['comp'] = COMP_ALGS['non-priv']
        job['epsilon'] = -1
        job['run'] = run
        jobs.append(job)
    return jobs, ga_objects

def priv_local_exp(dataset='Adult', problem='LR', runs=100):
    jobs = [] 
    print('Preping Priv Local Experiment {}'.format(dataset))
    directory = 'Priv_Local/{}/{}/'.format(dataset,problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    util_obj, _ = read_data_get_utility_obj(dataset, problem, vanilla_loss=True)

    for generation in [None, 10, 50, 100]:
        string = 0 if generation is None else generation
        params = copy.deepcopy(DEFAULT_PRIV_LOCAL)
        params['filename'] = directory + 'PrivLocal_'+str(string)
        params['gen_overide'] = generation
        for eps in GENS_EPS:
            for run in range(runs):
                job = params.copy()
                job['epsilon'] = eps 
                job['run'] = run
                jobs.append(job)

    return jobs, util_obj

def priv_local_mod_exp(dataset='Adult', problem='LR', runs=100):
    jobs = [] 
    print('Preping Priv Local MOD Experiment {}'.format(dataset))
    directory = 'Priv_Local_MOD/{}/{}/'.format(dataset,problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    util_obj, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=False)
    generation=100
    string = 0 if generation is None else generation
    params = copy.deepcopy(DEFAULT_PRIV_LOCAL)
    params['filename'] = directory + 'PrivLocal_'+str(string)
    params['gen_overide'] = generation
    params['delta'] = delta
    params['adv_comp'] = True
    for eps in GENS_EPS:
        for run in range(runs):
            job = params.copy()
            job['epsilon'] = eps 
            job['run'] = run
            jobs.append(job)
    return jobs, util_obj

def gupta_local_exp(dataset='Adult', problem = 'KM_16', runs=30):
    jobs = [] 
    print('Preping Gupta Local Search Experiment {}'.format(dataset))
    directory = 'Gupta_Local_Final/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    util_obj, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=True)

    params = copy.deepcopy(DEFAULT_GUPTA)
    params['filename'] = directory + 'GuptaLocal'
    for eps in GENS_EPS:
        for run in range(runs):
            job = params.copy()
            job['epsilon'] = eps 
            job['run'] = run
            jobs.append(job)
    return jobs, util_obj

def gupta_local_mod_exp(dataset='Adult', problem = 'KM_4', runs=30):
    jobs = [] 
    print('Preping Gupta Local Search MOD Experiment {}'.format(dataset))
    directory = 'Gupta_Local_Final_MOD/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    util_obj, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=True) 

    params = copy.deepcopy(DEFAULT_GUPTA)
    params['filename'] = directory + 'GuptaLocal'
    params['adv_comp'] = True 
    params['delta'] = delta
    for eps in GENS_EPS:
        for run in range(runs):
            job = params.copy()
            job['epsilon'] = eps 
            job['run'] = run
            jobs.append(job)
    return jobs, util_obj


def jones_exp(dataset='Adult', problem='KM_16', runs=70):
    jobs = [] 
    print('Preping Jones Experiment {}'.format(dataset))
    directory = 'Jones/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    data_dict = pickle.load(open('processed_datasets/' + dataset + '.p', 'rb'))
    util_obj, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=True)

    for alpha in [0.2, 0.6, 0.99]:
        params = copy.deepcopy(DEFAULT_JONES)
        params['filename'] = directory + 'jones_'+str(alpha)
        params['alpha'] = alpha
        params['delta'] = delta
        for eps in GENS_EPS:
            for run in range(30,100,1):
                job = params.copy()
                job['epsilon'] = eps 
                job['run'] = run
                jobs.append(job)
    return jobs, [util_obj, data_dict['clustering_sets']['public']]

def priv_gene_exp(dataset='Adult', problem='LR', runs=100):
    jobs = [] 
    print('Preping PrivGene Experiment {} {}'.format(dataset, problem))
    directory = 'PrivGene_Experiment/{}/{}/'.format(dataset, problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    ga_object, _ = read_data_get_utility_obj(dataset, problem, vanilla_loss=True)

    for generation in [None, 10, 50, 100]:
        string = 0 if generation is None else generation
        params = copy.deepcopy(DEFAULT_PRIV_GENE)
        params['filename'] = directory + 'PrivGene_std_' + str(string)
        params['gen_overide'] = generation
        for eps in GENS_EPS:
            for run in range(runs):
                job = params.copy()
                job['epsilon'] = eps 
                job['run'] = run
                jobs.append(job)
    #Non-private runs
    generation = 150
    params = copy.deepcopy(DEFAULT_PRIV_GENE)
    params['filename'] = directory + 'PrivGene_std_' + str(0)
    params['gen_overide'] = generation
    params['epsilon'] = -1
    params['non_priv'] = True
    for run in range(runs):
        job = params.copy()
        job['run'] = run
        jobs.append(job)
    return jobs, ga_object

def priv_eem_exp(dataset='Adult', problem='LR', runs=100):
    jobs = [] 
    print('Preping Priv EEM Experiment {}'.format(dataset))
    directory = 'Priv_eem/{}/{}/'.format(dataset,problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    util_obj, _ = read_data_get_utility_obj(dataset, problem, vanilla_loss=True)

    for generation in [None, 10, 50, 100]:
        string = 0 if generation is None else generation
        params = copy.deepcopy(DEFAULT_PRIV_EEM)
        params['filename'] = directory + 'PrivLocal_'+str(string)
        params['gen_overide'] = generation
        for eps in GENS_EPS:
            for run in range(runs):
                job = params.copy()
                job['epsilon'] = eps 
                job['run'] = run
                jobs.append(job)
    return jobs, util_obj

def priv_eem_mod_exp(dataset='Adult', problem='LR', runs=100):
    jobs = [] 
    print('Preping Priv EEM MOD Experiment {}'.format(dataset))
    directory = 'Priv_eem_MOD/{}/{}/'.format(dataset,problem)
    if not os.path.exists('Results/' + directory):
        os.makedirs('Results/' + directory)

    util_obj, delta = read_data_get_utility_obj(dataset, problem, vanilla_loss=False)
    generation=100
    string = 0 if generation is None else generation
    params = copy.deepcopy(DEFAULT_PRIV_EEM)
    params['filename'] = directory + 'PrivEEM_'+str(string)
    params['gen_overide'] = generation
    params['delta'] = delta
    params['adv_comp'] = True
    params['problem']=problem
    for eps in GENS_EPS:
        for run in range(runs):
            job = params.copy()
            job['epsilon'] = eps 
            job['run'] = run
            jobs.append(job)
    return jobs, util_obj

def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])

def main(args):
    
    ray.init(num_cpus=args.num_cpus)
    args_map = {
        'generations' : [generations_experiment, saving_GA],
        'pg_generations' : [PrivGene_generations_experiment, std_priv_gene],
        'selection' : [selection_experiment, saving_GA],
        'gupta':[gupta_local_exp, gupta_algo],
        'gupta_mod':[gupta_local_mod_exp, gupta_algo],
        'privGene' : [priv_gene_exp, std_priv_gene],
        'privEEM' : [priv_eem_exp, priv_eem],
        'privEEM_mod' : [priv_eem_mod_exp, priv_eem],
        'jones' : [jones_exp, jones_algo],
        'privLocal' : [priv_local_exp, priv_local],
        'privLocal_mod' : [priv_local_mod_exp, priv_local],
        'utility' : [utility_experiment, saving_GA],
        'grid' : [grid_experiment, saving_GA],
        'grid_rerun' : [grid_rerun, saving_GA],
        'non-priv_grid' : [non_priv_grid_experiment, saving_GA]
    }

    try:
        functions = args_map[args.experiment]
    except KeyError:
        print('Invalid Algorithm! Exiting')
        return None

    jobs, util = functions[0](dataset=args.dataset_name, problem=args.problem_name)
    util_obj = ray.put(util)
    function_r = ray.remote(functions[1])
    test = [function_r.remote(job, util_obj) for job in jobs]
    for _ in tqdm.tqdm(to_iterator(test), total=len(test)):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='')
    parser.add_argument('--dataset_name', type=str, default='Spam')
    parser.add_argument('--problem_name', type=str, default='KM_16')
    parser.add_argument('--num_cpus', type=int, default=4)
    args = parser.parse_args()
    print(vars(args))
    main(args)