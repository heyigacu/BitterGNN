import rdkit 
from rdkit.Chem import PandasTools
import pandas 
from rdkit import Chem
import os 

dir = os.path.abspath(os.path.dirname(__file__))

def drop_error_mols(path=dir+"/classification/N6512.csv"):
    df = pandas.read_csv(path, sep="\t", header=0)
    none_list=[]
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['Smiles'][i]) is None:
            none_list.append(i)
    num_error_molecules = len(none_list)
    print("dropping {} unidentified molecules".format(num_error_molecules))
    df=df.drop(none_list)
    df.to_csv(path, header=True, sep="\t", index=False)

# files = os.listdir(dir+"/classification/")
# for file in files:
#     drop_error_mols(dir+"/classification/"+file)

files = os.listdir(dir+"/regression/")
for file in files:
    drop_error_mols(dir+"/regression/"+file)
