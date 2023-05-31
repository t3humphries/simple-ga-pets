import numpy as np
import os
import scipy.stats
import pickle
import copy

from dict_helpers import *


#https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h

'''Returns a dictionary of results for all files'''
def file_reader(dir, filename, path = '/Users/data'): #TODO Redacted for blind review
    results_path = path + dir
    prefix = ''
    suffix = '.p'
    result = {}
    count = 0
    for file in os.scandir(results_path):
        if file.name.startswith(prefix + filename + '_') and file.name.endswith(suffix):
            tokens = file.name[len(prefix + filename + '_'):-len(suffix)].split('_')
            with open(os.path.join(results_path, file.name), 'rb') as input_file:
                value = pickle.load(input_file)
            set_dict_entry(result, tokens, value)
            count += 1
    print("Loaded {:d} files".format(count))
    return result

def read_specific_files(dir, filename, list_keys, runs=10, path='/Users/data'): #Redacted for blind review
    results_path = path + dir
    prefix = ''
    suffix = '.p'
    result = {}
    count = 0
    start = results_path+'/'+filename+'_'
    for keys in list_keys:
        args = '_'.join(keys)
        for r in range(runs):
            full_filename = start+args+ '_{}'.format(r)+suffix
            with open(full_filename, 'rb') as input_file:
                    value = pickle.load(input_file)
            set_dict_entry(result, keys+['{}'.format(r)], value)
            count += 1
    print("Loaded {:d} files".format(count))
    return result


'''Checking if a given list of keys meets the restriction
Null entries represent no restriction'''
def keys_compare(keys, restriction):
    to_return = True
    for i, restrict in enumerate(restriction):
        if restrict is not None:
            if keys[i] != restrict:
                to_return = False
    return to_return

''' Returns a deep copy of result only after applying the restriction
Retriction is either a key value to include or None to get all in that column'''
def extract_subest(result, restriction):
    to_return = {}
    for val in dict_iterator(result):
        if keys_compare(val[0], restriction):
            set_dict_entry(to_return, val[0], val[1])
    return to_return

''' Returns a nested dictionary with the same keys but the value is the loss computed by utility_obj
performance = True uses the max candidate found by the GA
performace = False uses the actual result chosen by the GA '''
def compute_final_utilities(result, utility_obj, performance=False):
    all_values = list(dict_iterator(result))
    if performance:
        candidates = [val[1][0][-1] for val in all_values]
    else:
        candidates = [val[1][2] for val in all_values]
    utils = utility_obj.get_utility_all(candidates)
    
    to_return = {}
    for i, value in enumerate(all_values):
        set_dict_entry(to_return, value[0], utils[i])
    return to_return

''' Returns a nested dictionary with the same keys but the value is the loss computed by utility_obj '''
def compute_final_utilities_simple(result, utility_obj):
    all_values = list(dict_iterator(result))
    candidates = [val[1] for val in all_values]
    utils = utility_obj.get_utility_all(candidates)
    to_return = {}
    for i, value in enumerate(all_values):
        set_dict_entry(to_return, value[0], utils[i])
    return to_return

'''Returns a dict with the avg perfomance array extracted'''
def get_avg_training_utilities_CI(result):
    to_return = {}
    depth = depth_dict(result)
    for val in dict_limited_iterator(result, depth-1):
        util_matrix = []
        for x in val[1].values():
            util_matrix.append(x[1])
        util_matrix = map(list, zip(*util_matrix))
        CI_list = list(zip(*[mean_confidence_interval(values) for values in util_matrix]))
        set_dict_entry(to_return, val[0], CI_list)
    return to_return

'''Returns a dict with the max perfomance array extracted 
utility is calculated using the given utility obj'''
def get_max_training_utilities_CI(result, utility_obj):
    to_return = {}
    depth = depth_dict(result)
    for diction in dict_limited_iterator(result, depth-1):
        util_matrix = []
        for val in diction[1].values():
            util_matrix.append(utility_obj.get_utility_all(val[0]))
        util_matrix = map(list, zip(*util_matrix))
        CI_list = list(zip(*[mean_confidence_interval(values) for values in util_matrix]))
        set_dict_entry(to_return, diction[0], CI_list)
    return to_return


'''Takes a dictionary of results and averages/computes confidence interval over runs dimension'''
def get_CI(dictionary):
    depth = depth_dict(dictionary)
    to_return = {}
    for val in dict_limited_iterator(dictionary, depth - 1):
        utilities = list(val[1].values())
        set_dict_entry(to_return, val[0], mean_confidence_interval(utilities))
    return to_return

'''Returns a list of k, [key, Confidence_iterval] pairs from a given CI dictionary'''
def get_top_k(result, k = 10, restriction=None):
    modified_dict = copy.deepcopy(result)
    if restriction is not None:
        modified_dict = extract_subest(modified_dict, restriction)
    to_return = []
    for _ in range(k):
        max_value = -np.inf
        max_list = []
        for val in dict_iterator(modified_dict):
            if val[1][0] > max_value:
                max_list = val
                max_value = val[1][0]
        to_return.append(max_list)
        del_dict_entry(modified_dict, max_list[0])
    return to_return

''' Reads files, outputs a pickled dictionary of utilites'''
def get_files_and_dump(utility_obj, in_dir, in_filename, out_filename):
    dictionary = file_reader('Adult/'+in_dir, in_filename)
    util = compute_final_utilities(dictionary, utility_obj, performance=False)
    with open('Utilitiy_Dictionaries/'+out_filename+'.p', 'wb') as output_file:
        pickle.dump(util, output_file)
    del util
    util_perf = compute_final_utilities(dictionary, utility_obj, performance=True)
    with open('Utilitiy_Dictionaries/'+out_filename+'_perf.p', 'wb') as output_file:
        pickle.dump(util_perf, output_file)
    del util_perf
    del dictionary

''' Reads files, returns dictionary of utilites'''
def get_files(utility_obj, in_dir, in_filename):
    dictionary = file_reader('Adult/'+in_dir, in_filename)
    util = compute_final_utilities(dictionary, utility_obj, performance=False)
    util_perf = compute_final_utilities(dictionary, utility_obj, performance=True)
    return util, util_perf