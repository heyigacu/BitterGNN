import torch
from rdkit import Chem
from dgllife.utils import mol_to_complete_graph
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer, WeaveEdgeFeaturizer, AttentiveFPBondFeaturizer
from dgllife.utils import *

def canonical_atom_featurlizer():
    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='h')
    n_nfeats = node_featurizer.feat_size('h')
    return node_featurizer, n_nfeats

def canonical_bond_featurlizer():
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='e', self_loop=False)
    n_efeats = edge_featurizer.feat_size('e')
    return edge_featurizer, n_efeats

def weave_bond_featurlizer():
    edge_featurizer = WeaveEdgeFeaturizer(edge_data_field='e', max_distance=7)
    n_efeats = edge_featurizer.feat_size()
    return edge_featurizer, n_efeats

def afp_bond_featurlizer():
    edge_featurizer = AttentiveFPBondFeaturizer(bond_data_field='e', self_loop=False)
    n_efeats = edge_featurizer.feat_size('e')
    return edge_featurizer, n_efeats


def featurize_atoms(mol):
    feats = []
    for atom in mol.GetAtoms():
        feats.append(atom.GetAtomicNum())
    return {'h': torch.tensor(feats).reshape(-1, 1).float()}


def featurize_bonds(mol):
    feats = []
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        btype = bond_types.index(bond.GetBondType())
        # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
        feats.extend([btype, btype])
    return {'e': torch.tensor(feats).reshape(-1, 1).float()}


def featurize_bonds1(mol):
    feats = []
    for bond in mol.GetBonds():
        # One bond between atom u and v corresponds to two edges (u, v) and (v, u)
        feats.extend([1, 1])
    return {'e': torch.tensor(feats).reshape(-1, 1).float()}


def atom_number_featurizer():
    return featurize_atoms, 1

def bond_number_featurizer():
    return featurize_bonds, 1

def bond_1_featurizer():
    return featurize_bonds1, 1



node_featurizer, n_nfeats = atom_number_featurizer()
edge_featurizer, n_efeats = bond_number_featurizer()

g = smiles_to_bigraph('Cn1c(=O)c2c(ncn2C)n(C)c1=O', node_featurizer=node_featurizer, edge_featurizer=edge_featurizer)
print(g.edata)








