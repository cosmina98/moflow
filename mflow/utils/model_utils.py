import torch
import numpy as np
from data.smile_to_graph import GGNNPreprocessor
from rdkit import Chem

from data import transform_qm9,transform_mydatasets
from data.transform_zinc250k import one_hot_zinc250k, transform_fn_zinc250k
from data.transform_mydatasets import one_hot, transform_fn

from mflow.models.model import MoFlow as Model


def load_model(snapshot_path, model_params, debug=False):
    print("loading snapshot: {}".format(snapshot_path))
    if debug:
        print("Hyper-parameters:")
        model_params.print()
    model = Model(model_params)

    device = torch.device('cpu')
    model.load_state_dict(torch.load(snapshot_path, map_location=device))
    return model


def smiles_to_adj(mol_smiles, data_name='qm9'):
    out_size = 9
    transform_fn = transform_qm9.transform_fn

    if data_name == 'zinc250k':
        out_size = 38
        transform_fn = transform_fn_zinc250k
    elif data_name in ['ames_train1_pos', 'ames_train1_neg','bbb_martins_train1_pos', 'bbb_martins_train1_neg','cyp1a2_veith_train1_pos','cyp1a2_veith_train1_neg', \
                   'cyp2c19_veith_train1_pos','cyp2c19_veith_train1_neg','herg_karim_train1_pos','herg_karim_train1_neg','lipophilicity_astrazeneca_train1_pos','lipophilicity_astrazeneca_train1_neg']:
        out_size = 152
        transform_fn = exec('transform_fn_{}'.format(data_name))

    preprocessor = GGNNPreprocessor(out_size=out_size, kekulize=True)
    canonical_smiles, mol = preprocessor.prepare_smiles_and_mol(Chem.MolFromSmiles(mol_smiles)) # newly added crucial important!!!
    atoms, adj = preprocessor.get_input_features(mol)
    atoms, adj, _ = transform_fn((atoms, adj, None))
    adj = np.expand_dims(adj, axis=0)
    atoms = np.expand_dims(atoms, axis=0)

    adj = torch.from_numpy(adj)
    atoms = torch.from_numpy(atoms)
    return adj, atoms


def get_latent_vec(model, mol_smiles, data_name='qm9'):
    adj, atoms = smiles_to_adj(mol_smiles, data_name)
    with torch.no_grad():
        z = model(adj, atoms)
    z = np.hstack([z[0][0].cpu().numpy(), z[0][1].cpu().numpy()]).squeeze(0) # change later !!! according to debug
    return z
