
import numpy as np
import dgl
from dgl import DGLGraph
import torch
from rdkit import Chem

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)
    if encode_unknown and (x not in allowable_set):
        x = None
    return list(map(lambda s: x == s, allowable_set))

def AtomNumAtomEmbedding(atom):
    return np.array(atom.GetAtomicNum())

def BlankBondEmbedding(bond):
    return np.array([1])

def BondTypeBondEmbedding(bond):
    """
    >>print(BondTypeBondEmbedding(Chem.MolFromSmiles("C=C").GetBondBetweenAtoms(0,1)))
    """
    bond_types = [Chem.rdchem.BondType.SINGLE, 
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE, 
                Chem.rdchem.BondType.AROMATIC]
    return np.array(bond_types.index(bond.GetBondType()))

def CanonicalAtomEmbedding(atom):
    """
    equal to CanonicalAtomFeaturizer(atom_data_field='h') in dglife
    >>print(CanonicalAtomEmbedding(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
    >>print(CanonicalAtomFeaturizer(atom_data_field='h')(Chem.MolFromSmiles("C")))
    """
    atom_features = one_hot_encoding(atom.GetSymbol(), 
                    ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                    'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                    'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb'], 
                    )
    atom_features += one_hot_encoding(atom.GetDegree(), list(range(11)))
    atom_features += one_hot_encoding(atom.GetImplicitValence(), list(range(7)))
    atom_features += [atom.GetFormalCharge()]
    atom_features += [atom.GetNumRadicalElectrons()]
    atom_features += one_hot_encoding(atom.GetHybridization(), 
                    [Chem.rdchem.HybridizationType.SP,
                    Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3,
                    Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2])
    atom_features += [atom.GetIsAromatic()]
    atom_features += one_hot_encoding(atom.GetTotalNumHs(), list(range(5)))
    return np.array(atom_features)

def CanonicalBondEmbedding(bond):
    """
    equal to CanonicalBondFeaturizer(bond_data_field='e', self_loop=False) in dglife
    >>print(CanonicalBondEmbedding(Chem.MolFromSmiles("CC").GetBondBetweenAtoms(0,1)))
    >>print(CanonicalBondFeaturizer(bond_data_field='e', self_loop=False)(Chem.MolFromSmiles("CC")))
    """
    bond_features = one_hot_encoding(bond.GetBondType(), 
                    [Chem.rdchem.BondType.SINGLE,
                    Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE,
                    Chem.rdchem.BondType.AROMATIC]
                    )
    bond_features += [bond.GetIsConjugated()]
    bond_features += [bond.IsInRing()]
    bond_features += one_hot_encoding(bond.GetStereo(), 
                    [Chem.rdchem.BondStereo.STEREONONE,
                    Chem.rdchem.BondStereo.STEREOANY,
                    Chem.rdchem.BondStereo.STEREOZ,
                    Chem.rdchem.BondStereo.STEREOE,
                    Chem.rdchem.BondStereo.STEREOCIS,
                    Chem.rdchem.BondStereo.STEREOTRANS])
    return np.array(bond_features)

# print(CanonicalAtomEmbedding(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
# print(CanonicalBondEmbedding(Chem.MolFromSmiles("C=C").GetBondBetweenAtoms(0,1)))

def Graph_smiles(smiles, 
                 get_atom_embedding=CanonicalAtomEmbedding, 
                 get_bond_embedding=CanonicalBondEmbedding):
    molecule = Chem.MolFromSmiles(smiles)
    g = DGLGraph()
    g.add_nodes(molecule.GetNumAtoms())
    node_features = []
    edge_features = []
    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i) 
        atom_i_features = get_atom_embedding(atom_i) 
        node_features.append(atom_i_features)
        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                g.add_edges(i,j) 
                bond_features_ij = get_bond_embedding(bond_ij) 
                edge_features.append(bond_features_ij)
    g.ndata['h'] = torch.from_numpy(np.array(node_features)).float()
    g.edata['e'] = torch.from_numpy(np.array(edge_features)).float()
    # g = dgl.add_self_loop(g, fill_data='sum')
    return g

