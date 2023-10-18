"""
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5608695/
AdaBoost Dagan-Wiener
"""
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.QED import properties
from rdkit.Chem.EState.EState import MinEStateIndex,MaxEStateIndex, EStateIndices
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers
from sklearn.model_selection import KFold
import joblib

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class AdaBoost(object):
    def __init__(self):
        pass
    @staticmethod
    def descriptros(index_filepath,qikprop_filepath):
        df = pd.read_csv(index_filepath, header=0, sep='\t') 
        indices = list(df['Idx'])
        smileses = list(df['Smiles'])
        trues = np.array((df['Label']),dtype=np.int64)
        df_q = pd.read_csv(qikprop_filepath, header=0, sep=',')
        features = np.array(df_q)[indices][:,1:-2]
        other_features=[]
        for smiles in smileses:
            mol = Chem.MolFromSmiles(smiles)
            qeds = list(properties(mol))
            qeds_need = [qeds[0],qeds[1],qeds[4],qeds[5],qeds[6]] #MW, ALOGP, PSA, ROTB, Ar ring 
            chirals = pow(len(tuple(EnumerateStereoisomers(mol))),0.5)
            mol_H = Chem.AddHs(mol)
            AllChem.ComputeGasteigerCharges(mol_H)
            contribs = [mol_H.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol_H.GetNumAtoms())]
            charge = sum(contribs) #Total charge.
            others = [mol.GetNumHeavyAtoms(), CalcNumRings(mol),sum(EStateIndices(mol)), charge, chirals, Chem.Crippen.MolMR(mol), Chem.Crippen.MolLogP(mol)] + qeds_need #Ring, HA, Chiral, MR, poloar...
            other_features.append(others)
        return np.concatenate((features,other_features), axis=1),trues
    @staticmethod
    def train(index_filepath,qikprop_filepath,save_name):
        x, y = AdaBoost.descriptros(index_filepath,qikprop_filepath)
        clf = AdaBoostClassifier(algorithm='SAMME.R',learning_rate=1.0, n_estimators=100, random_state=0)
        clf.fit(x, y)
        joblib.dump(clf, path+'/pretrained/{}.pkl'.format(save_name))
    @staticmethod
    def test(index_filepath,qikprop_filepath, save_name):
        features, trues = AdaBoost.descriptros(index_filepath,qikprop_filepath)
        model =  joblib.load(path+'/pretrained/{}.pkl'.format(save_name))
        preds = model.predict_proba(features)
        return trues,preds
    


