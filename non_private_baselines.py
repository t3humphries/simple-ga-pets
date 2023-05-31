import pickle
import numpy as np
from ga_util_objects import K_Median, Log_Regression

from pam_algo._k_medoids import K_Medoids_Custom
from sklearn.linear_model import LogisticRegression


def logistic_reg(dataset):
    data_dict = pickle.load(open('processed_datasets/' + dataset + '.p', 'rb'))
    test_util_obj = Log_Regression(data_dict['test_train_sets']['test_x'],
                                        data_dict['test_train_sets']['test_y'])
    sk_learn_models = []
    for run in range(100):
        clf = LogisticRegression(solver='sag').fit(data_dict['test_train_sets']['train_x'], data_dict['test_train_sets']['train_y'])
        sk_learn_models.append(np.concatenate([clf.coef_[0], clf.intercept_]))
    sk_learn_fit = [test_util_obj.get_utility(cand) for cand in sk_learn_models] 
    return np.mean(sk_learn_fit)

def k_medoids(dataset, k=4):
    data_dict = pickle.load(open('processed_datasets/' + dataset + '.p', 'rb'))
    public_data = data_dict['clustering_sets']['public']
    private_data = data_dict['clustering_sets']['private']
    ga_object = K_Median(public_data, private_data, k)
    kmediods = K_Medoids_Custom(init='build', n_clusters=k).fit(ga_object.distance_matrix)
    result = kmediods.medoid_indices_
    return ga_object.get_utility(result)

if __name__ == "__main__":
    utils = {}
    for dataset in ['Spam', 'Mushrooms', 'Adult', 'Credit']:
        utils[dataset+'KM_16'] = k_medoids(dataset, k=16)
        utils[dataset+'LR'] = logistic_reg(dataset)
    with open( 'misc_p_files/non_private_baselines.p', "wb" ) as outfile:
        pickle.dump(utils, outfile)
    