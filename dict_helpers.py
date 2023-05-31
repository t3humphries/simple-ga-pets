import operator
from functools import reduce 
#-------Helpers for nested dictionaries
def initialize_dict():
    return defaultdict(initialize_dict)

def get_dict_entry(dictionary, keys):
    return reduce(operator.getitem, keys, dictionary)

def update_dict_entry(dictionary, keys, value):
    get_dict_entry(dictionary, keys[:-1])[keys[-1]] = value

def set_dict_entry(dictionary, keys, value):
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value
    
def del_dict_entry(dictionary, keys):
    del get_dict_entry(dictionary, keys[:-1])[keys[-1]]

def dict_iterator(dictionary, keys=None):
    if keys is None:
        keys = []
    for key, value in dictionary.items():
        new_keys = keys + [key]
        if isinstance(value, dict):
            for pair in dict_iterator(value, keys=new_keys):
                yield pair
        else:
            yield [new_keys, value]

def depth_dict(dictionary):
    iteration = list(dict_iterator(dictionary))
    first_key = iteration[0][0]
    return len(first_key)

def dict_limited_iterator(dictionary, limit, keys=None):
    if keys is None:
        keys = []
    for key, value in dictionary.items():
        new_keys = keys + [key]
        if limit > 1:
            for pair in dict_limited_iterator(value, limit - 1, keys=new_keys):
                yield pair
        else:
            yield [new_keys, value]