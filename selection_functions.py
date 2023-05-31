import numpy as np
import copy

def normal_selection(pop_fit, chosen_size, epsilon, sensitivity, rng):
    return sorted(range(len(pop_fit)), key=lambda i: pop_fit[i])[-chosen_size:]

def gumbel_select(pop_fit, chosen_size, eps_per_sample, sensitivity, rng):
    noisy_fit = [x + rng.gumbel(scale=2 * sensitivity / eps_per_sample) for x in pop_fit]
    sorted_indicies = sorted(range(len(noisy_fit)), key=lambda i: noisy_fit[i])
    chosen_indicies = sorted_indicies[-chosen_size:]
    return chosen_indicies

def noisy_max_select(pop_fit, chosen_size, eps_per_sample, sensitivity, rng):
    chosen_indicies = []
    for _ in range(chosen_size):
        noisy_fit = np.array([cand + rng.laplace(scale= 2 * sensitivity / eps_per_sample) for cand in pop_fit])
        noisy_fit[chosen_indicies] = -np.inf #Do not consider already chosen ones
        chosen = np.argmax(noisy_fit)
        chosen_indicies.append(chosen)
    return chosen_indicies

def exp_noisy_max_select(pop_fit, chosen_size, eps_per_sample, sensitivity, rng):
    chosen_indicies = []
    for _ in range(chosen_size):
        noisy_fit = np.array([cand + rng.exponential(scale= 2 * sensitivity / eps_per_sample) for cand in pop_fit])
        noisy_fit[chosen_indicies] = -np.inf #Do not consider already chosen ones
        chosen = np.argmax(noisy_fit)
        chosen_indicies.append(chosen)
    return chosen_indicies

def permute_flip_select(pop_fit, chosen_size, eps_per_sample, sensitivity, rng):
    chosen_indicies = []
    utils = copy.deepcopy(pop_fit)
    for _ in range(chosen_size):
        r = permute_and_flip(utils, eps_per_sample, sensitivity, rng)
        chosen_indicies.append(r)
        utils[r] = -np.inf #Do not consider already chosen candidates
    return chosen_indicies

'''Algorithm 1 from Mckenna and Sheldon'''
def permute_and_flip(utilities, eps, sensitivity, rng):
    q_star = np.max(utilities)
    for r in rng.permutation(len(utilities)):
        p_r = np.exp(eps/(2 * sensitivity) * (utilities[r] - q_star))
        if rng.uniform() <= p_r:
            return r
    raise RuntimeError('Theoretically impossible error in permute and flip')
