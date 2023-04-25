import json

import numpy as np

atomic_num_list = [1,3,5,6, 7, 8, 9, 11,14,15, 16, 17,19,20,24,25,26,28,29,30, 33,34,35,44, 50,51,53,78,79,80, 0]  # 0 is for virtual node.
max_atoms = 150
n_bonds = 4


def one_hot(data, out_size=150):
    num_max_id = len(atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b


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