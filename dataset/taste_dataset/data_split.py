import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import os 


pwd = os.path.abspath(os.path.dirname(__file__))
print(pwd)

files = ["Bitter", "Sweet", "Sour", "Salt", "Umami", "Kokumi", "Astringent", "Tasteless"]



for i,file in enumerate(files):
    path = pwd + "/multi_flavor/" + file + ".csv"
    if i == 0:
        df = pd.read_csv(path, header=0, sep="\t")
        df.insert(4,"Label",str(i))
    else:
        temp = pd.read_csv(path, header=0, sep="\t")
        temp.insert(4,"Label",str(i))
        df = pd.concat([df,temp])
print(df["Label"].value_counts())

def RandomSample4Multi(df):
    balanced_df = df.query('Label=="5"')
    num = len(balanced_df)
    print(num)
    names = files
    for name in names:
        if name != 'Kokumi':
            df_ = df.query('Taste==@name')
            df_ = shuffle(df_)
            df_ = df_.sample(frac=num/len(df_), random_state=1, replace=True)
            balanced_df = pd.concat([balanced_df, df_])
    return balanced_df
balanced_df = RandomSample4Multi(df)
print(balanced_df["Label"].value_counts())
balanced_df.to_csv(pwd+"/multi_flavor/sampled.csv",header=True, index=False, sep="\t")

# df = pd.read_csv("/home/hy/Documents/Project/bitterants/dataset/taste_dataset/bitter_nonbitter/bn_test.csv", sep=',',header=0)
# df.to_csv("/home/hy/Documents/Project/bitterants/dataset/taste_dataset/bitter_nonbitter/bn_test.csv", sep='\t',header=0)

# df = pd.read_csv("/home/hy/Documents/Project/bitterants/dataset/taste_dataset/bitter_nonbitter/bn_train_val.csv", sep=',',header=0)
# df.to_csv("/home/hy/Documents/Project/bitterants/dataset/taste_dataset/bitter_nonbitter/bn_train_val.csv", sep='\t',header=0)

# df = pd.read_csv("/home/hy/Documents/Project/bitterants/dataset/taste_dataset/bitter_sweet/bs_test.csv", sep=',',header=0)
# df.to_csv("/home/hy/Documents/Project/bitterants/dataset/taste_dataset/bitter_sweet/bs_test.csv", sep='\t',header=0)

# df = pd.read_csv("/home/hy/Documents/Project/bitterants/dataset/taste_dataset/bitter_sweet/bs_train_val.csv", sep=',',header=0)
# df.to_csv("/home/hy/Documents/Project/bitterants/dataset/taste_dataset/bitter_sweet/bs_train_val.csv", sep='\t',header=0)