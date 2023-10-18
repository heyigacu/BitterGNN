import pandas as pd

import os
import numpy as np
pwd = os.path.dirname(os.path.dirname(__file__))

def get_result():
    df = pd.read_csv(pwd+"/contrast_predictors/BitterX_result.csv", header=None, sep="\t")
    df.columns = ["JobName", "Label"]
    return np.array(list(df["Label"]))

