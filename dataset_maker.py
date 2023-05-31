import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def import_dataset(filename):
    dataframe = pd.read_csv(filename)
    return dataframe

def encode_dataset(dataframe, binary_encoding=True, class_name='class', permuatation=None):
    # convert class to ints
    dataframe[class_name] = dataframe[class_name].astype('category')
    dataframe[class_name] = dataframe[class_name].cat.codes
    # Split off the labels
    labels = dataframe[class_name].values
    dataframe.drop(columns=class_name, inplace=True)

    # get categorical attributes
    columns = dataframe.columns
    numeric_columns = dataframe._get_numeric_data().columns
    if permuatation is None:
        categorical_attributes = list(set(columns) - set(numeric_columns))
    else:
        categorical_attributes = permuatation

    if binary_encoding:
        #print('Converting categorial to numeric using a one hot encoding')
        binary_attributes = []
        other_attributes = []
        for attribute in categorical_attributes:
            if len(dataframe[attribute].unique()) == 2:
                binary_attributes.append(attribute)
            else:
                other_attributes.append(attribute)

        encoded_dataframe = pd.get_dummies(dataframe, columns=other_attributes)
        encoded_dataframe[binary_attributes] = encoded_dataframe[binary_attributes].astype(
            'category')
        encoded_dataframe[binary_attributes] = encoded_dataframe[binary_attributes].apply(
            lambda x: x.cat.codes)
    else:
        print('Converting categorial to numeric using an integer encoding')
        encoded_dataframe = dataframe.copy()
        encoded_dataframe[categorical_attributes] = encoded_dataframe[categorical_attributes].astype(
            'category')
        encoded_dataframe[categorical_attributes] = encoded_dataframe[categorical_attributes].apply(
            lambda x: x.cat.codes)
    return encoded_dataframe, labels

'''(0,1) min max normalization'''
def normalize_dataset(dataframe):
    features = dataframe.values
    scaler = MinMaxScaler()
    return scaler.fit_transform(features)

'''z-score smoothing'''
def scale_dataset(dataframe):
    features = dataframe.values
    return preprocessing.scale(features)

def create_rand_public(shuffled_features):
    binary_cols = [len(np.unique(shuffled_features[:,i])) == 2 for i in range(len(shuffled_features[0]))]
    public_size = int(len(shuffled_features) * (2/8))
    public_data = []
    for _ in range(public_size):
        data_point = [np.random.randint(low=0, high=2) if x else np.random.rand() for x in binary_cols]
        public_data.append(data_point)
    public_data = np.array(public_data)
    return public_data

def dataset_to_pickle(dataset, filename, classification='class'):
    output = {}

    np.random.seed(10000019)
    features, labels = encode_dataset(dataset, class_name=classification)
    shuffled_idx = np.random.permutation(len(features))
    
    shuffled_labels = labels[shuffled_idx]
    output['labels'] = shuffled_labels
    normal_data_min_max = normalize_dataset(features) 
    shuffled_features = normal_data_min_max[shuffled_idx]
    output['normal_data_min_max'] = shuffled_features

    output['clustering_sets'] = {}
    public_data = create_rand_public(shuffled_features) 
    output['clustering_sets']['public'] = public_data
    output['clustering_sets']['private'] = shuffled_features
    print(len(public_data), len(shuffled_features))

    output['test_train_sets']={}
    training_size = int(normal_data_min_max.shape[0] * 0.8)

    output['test_train_sets']['train_x'] = shuffled_features[:training_size]
    output['test_train_sets']['train_y'] = shuffled_labels[:training_size]
    output['test_train_sets']['test_x'] = shuffled_features[training_size:]
    output['test_train_sets']['test_y'] = shuffled_labels[training_size:]

    print(len(output['test_train_sets']['train_y']), len(output['test_train_sets']['test_y']))
    pickle.dump(output, open( filename+'.p', "wb" ))
    return None

if __name__ == "__main__":
    output_dir = 'processed_datasets/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Adult
    dataframe = import_dataset('Datasets/dataset_adult_no_missing.csv')
    dataset_to_pickle(dataframe, output_dir + 'Adult')
    #Credit
    dataframe = import_dataset('Datasets/credit.csv')
    dataset_to_pickle(dataframe, output_dir + 'Credit', classification='default payment next month')
    #Mushrooms
    dataframe = import_dataset('Datasets/mushrooms_no_missing.csv')
    dataset_to_pickle(dataframe, output_dir + 'Mushrooms')
    #Spam
    dataframe = import_dataset('Datasets/spam.csv')
    dataset_to_pickle(dataframe, output_dir + 'Spam')