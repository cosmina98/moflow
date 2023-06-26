import json

import numpy as np
atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 78, 20, 24, 25, 26, 29, 30, 33, 34, 35, 50, 51, 53,0]
max_atoms = 133
n_bonds = 4


def one_hot(data, out_size=max_atoms):
    num_max_id = len(atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b

def get_atomic_num_list(data_name):
    if  data_name == 'ames_25_train1_neg':
        atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_25_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_33_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_33_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_50_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_50_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_40_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'ames_40_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'bbb_martins_25_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 11, 15, 16, 17, 53,0]
    elif data_name == 'bbb_martins_25_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 35, 53,0]
    elif data_name == 'bbb_martins_33_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 11, 15, 16, 17,0]
    elif data_name == 'bbb_martins_33_train1_pos':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 11, 15, 16, 17,0]
    elif data_name == 'bbb_martins_50_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 11, 15, 16, 17, 53,0]
    elif data_name == 'bbb_martins_50_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 35, 53,0]
    elif data_name == 'bbb_martins_40_train1_neg':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 11, 15, 16, 17, 53,0]
    elif data_name == 'bbb_martins_40_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 15, 16, 17, 20, 35, 53,0]     
    elif data_name == 'cyp1a2_veith_25_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 14, 15, 16, 17, 78, 19, 25, 26, 29, 30, 33, 34, 35, 50, 51, 53,0]
    elif data_name == 'cyp1a2_veith_25_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 28, 29, 34, 35, 53,0]
    elif data_name == 'cyp1a2_veith_33_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 19, 78, 25, 26, 27, 30, 33, 34, 35, 50, 51, 53,0]
    elif data_name == 'cyp1a2_veith_33_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 29, 34, 35, 53,0]
    elif data_name == 'cyp1a2_veith_40_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 19, 14, 24, 25, 26, 27, 30, 33, 34, 35, 51, 53,0]
    elif data_name == 'cyp1a2_veith_40_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 14, 29, 34, 35, 53,0]
    elif data_name == 'cyp1a2_veith_50_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 19, 14, 24, 25, 26, 27, 30, 33, 34, 35, 51, 53,0]
    elif data_name == 'cyp1a2_veith_50_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 80, 28, 29, 34, 35, 53,0]
    elif data_name == 'cyp2c19_veith_25_train1_neg':
         atomic_num_list=[1, 3, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 78, 20, 26, 29, 33, 35, 50, 51, 53,0]
    elif data_name == 'cyp2c19_veith_25_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 19, 29, 35, 53,0]
    elif data_name == 'cyp2c19_veith_33_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 80, 19, 28, 29, 35, 53,0]
    elif data_name == 'cyp2c19_veith_33_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 19, 20, 80, 25, 26, 29, 30, 33, 34, 35, 44, 50, 51, 53,0]
    elif data_name == 'cyp2c19_veith_40_train1_neg':
         atomic_num_list=[1, 3, 5, 6, 7, 8, 9, 11, 78, 15, 16, 17, 14, 80, 20, 25, 26, 29, 30, 33, 35, 50, 51, 53,0]
    elif data_name == 'cyp2c19_veith_40_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 19, 78, 26, 29, 35, 53,0]
    elif data_name == 'cyp2c19_veith_50_train1_neg':
         atomic_num_list=[1, 3, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 78, 20, 24, 25, 26, 29, 30, 33, 34, 35, 50, 51, 53,0]
    elif data_name == 'cyp2c19_veith_50_train1_pos':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 80, 19, 78, 29, 35, 53,0]
    elif data_name == 'herg_karim_25_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 79, 16, 17, 15, 34, 35, 53,0]
    elif data_name == 'herg_karim_25_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'herg_karim_33_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53,0]
    elif data_name == 'herg_karim_33_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 16, 17, 53,0]
    elif data_name == 'herg_karim_40_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 34, 35, 53,0]
    elif data_name == 'herg_karim_40_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17, 53,0]
    elif data_name == 'herg_karim_50_train1_neg':
         atomic_num_list=[1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 79, 34, 35, 53,0]
    elif data_name == 'herg_karim_50_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 14, 16, 17, 53,0]
    elif data_name == 'lipophilicity_astrazeneca_25_train1_neg':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 16, 17, 53,0]
    elif data_name == 'lipophilicity_astrazeneca_25_train1_pos':
         atomic_num_list=[1, 35, 6, 7, 8, 9, 15, 16, 17,0]
    elif data_name == 'lipophilicity_astrazeneca_33_train1_neg':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 16, 17,0]
    elif data_name == 'lipophilicity_astrazeneca_33_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53,0]
    elif data_name == 'lipophilicity_astrazeneca_40_train1_neg':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 16, 17, 53,0]
    elif data_name == 'lipophilicity_astrazeneca_40_train1_pos':
         atomic_num_list=[1, 34, 35, 5, 6, 7, 8, 9, 15, 16, 17,0]
    elif data_name == 'lipophilicity_astrazeneca_50_train1_neg':
         atomic_num_list=[1, 35, 5, 6, 7, 8, 9, 15, 16, 17,0]
    elif data_name == 'lipophilicity_astrazeneca_50_train1_pos':
         atomic_num_list=[1, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53,0]
          
    return atomic_num_list

def get_max_atoms(data_name):
    if  data_name == 'ames_25_train1_neg':
        max_atoms=57
    elif data_name == 'ames_25_train1_pos':
         max_atoms=57
    elif data_name == 'ames_33_train1_neg':
         max_atoms=57
    elif data_name == 'ames_33_train1_pos':
         max_atoms=57
    elif data_name == 'ames_50_train1_neg':
         max_atoms=57
    elif data_name == 'ames_50_train1_pos':
         max_atoms=57
    elif data_name == 'ames_40_train1_neg':
         max_atoms=57
    elif data_name == 'ames_40_train1_pos':
         max_atoms=57
    elif data_name == 'bbb_martins_25_train1_neg':
         max_atoms=133
    elif data_name == 'bbb_martins_25_train1_pos':
         max_atoms=76
    elif data_name == 'bbb_martins_33_train1_neg':
         max_atoms=133
    elif data_name == 'bbb_martins_33_train1_pos':
         max_atoms=76
    elif data_name == 'bbb_martins_50_train1_neg':
         max_atoms=133
    elif data_name == 'bbb_martins_50_train1_pos':
         max_atoms=76
    elif data_name == 'bbb_martins_40_train1_neg':
         max_atoms=133
    elif data_name == 'bbb_martins_40_train1_pos':
         max_atoms=76
    elif data_name == 'cyp1a2_veith_25_train1_neg':
         max_atoms=133
    elif data_name == 'cyp1a2_veith_25_train1_pos':
         max_atoms=133
    elif data_name == 'cyp1a2_veith_33_train1_neg':
         max_atoms=133
    elif data_name == 'cyp1a2_veith_33_train1_pos':
         max_atoms=133
    elif data_name == 'cyp1a2_veith_50_train1_neg':
         max_atoms=133
    elif data_name == 'cyp1a2_veith_50_train1_pos':
         max_atoms=133
    elif data_name == 'cyp1a2_veith_40_train1_neg':
         max_atoms=133
    elif data_name == 'cyp1a2_veith_40_train1_pos':
         max_atoms=133
    elif data_name == 'cyp2c19_veith_25_train1_neg':
         max_atoms=133
    elif data_name == 'cyp2c19_veith_25_train1_pos':
         max_atoms=133
    elif data_name == 'cyp2c19_veith_33_train1_neg':
         max_atoms=101
    elif data_name == 'cyp2c19_veith_33_train1_pos':
         max_atoms=67
    elif data_name == 'cyp2c19_veith_50_train1_neg':
         max_atoms=101
    elif data_name == 'cyp2c19_veith_50_train1_pos':
         max_atoms=106
    elif data_name == 'cyp2c19_veith_40_train1_neg':
         max_atoms=114
    elif data_name == 'cyp2c19_veith_40_train1_pos':
         max_atoms=106
    elif data_name == 'herg_karim_25_train1_neg':
         max_atoms=58
    elif data_name == 'herg_karim_25_train1_pos':
         max_atoms=50
    elif data_name == 'herg_karim_33_train1_neg':
         max_atoms=58
    elif data_name == 'herg_karim_33_train1_pos':
         max_atoms=50
    elif data_name == 'herg_karim_50_train1_neg':
         max_atoms=58
    elif data_name == 'herg_karim_50_train1_pos':
         max_atoms=50
    elif data_name == 'herg_karim_40_train1_neg':
         max_atoms=58
    elif data_name == 'herg_karim_40_train1_pos':
         max_atoms=50
    elif data_name == 'lipophilicity_astrazeneca_25_train1_neg':
         max_atoms=115
    elif data_name == 'lipophilicity_astrazeneca_25_train1_pos':
         max_atoms=72
    elif data_name == 'lipophilicity_astrazeneca_33_train1_neg':
         max_atoms=65
    elif data_name == 'lipophilicity_astrazeneca_33_train1_pos':
         max_atoms=72
    elif data_name == 'lipophilicity_astrazeneca_50_train1_neg':
         max_atoms=115
    elif data_name == 'lipophilicity_astrazeneca_50_train1_pos':
         max_atoms=47
    elif data_name == 'lipophilicity_astrazeneca_40_train1_neg':
         max_atoms=115
    elif data_name == 'lipophilicity_astrazeneca_40_train1_pos':
         max_atoms=61


    return max_atoms


def transform_fn(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label


def get_val_ids(data_name):
    file_path = '../data/valid_idx_{}.json'.format(data_name)
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [int(idx)-1 for idx in data['valid_idxs']]
    return val_ids