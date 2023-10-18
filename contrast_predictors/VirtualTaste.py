import os
import numpy as np
import pandas as pd
pwd = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

df = pd.read_csv(pwd+"/dataset/taste_dataset/bitter_sweet/bs_test.csv", header=0, sep="\t")
df['Smiles'].to_csv(pwd+"/contrast_predictors/virtual_taste_test.dat", index=False, header=False)

def get_virtual_taste_result():
    with open(pwd+"/contrast_predictors/VirtualTasteResult.csv", "r") as f:
        lines = f.readlines()
        ls = []
        for line in lines:
            if line.startswith("Bitter"):
                if line.split(",")[2] == "0":
                    ls.append([float(line.split(",")[3].strip()), 1-float(line.split(",")[3].strip())])
                elif line.split(",")[2] == "1":
                    ls.append([1-float(line.split(",")[3].strip()), float(line.split(",")[3].strip())])
                else:
                    print("error")
        return np.array(ls)
print(get_virtual_taste_result())
