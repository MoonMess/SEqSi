import numpy as np
from argparse import ArgumentParser
from os.path import join
from glob import glob
import pandas as pd


EXPERIMENT = "augmentation" 
#EXPERIMENT = "artifacts"

if EXPERIMENT == "augmentation":
    alter_names_clean = [
        "Original",
        "Shift",
        "Scale ($<1$)",
        "Scale ($>1$)",
        "Affine",
        "Shift saturated",
        "Scale ($>1$) saturated",
        "Affine saturated",
        "Noise low",
        "Noise high",
        "Gamma (darken)",
        "Gamma (lighten)"
    ]
if EXPERIMENT == "artifacts":
    alter_names_clean = [
        "Arti. (1.5)",
        "Arti. low",
        "Arti. medium",
        "Arti. high",
    ]


exp_names = ["none", "Aff", "Naff", "All"]
met_names = ["ordinary_BCE.npz", "SEq_BCE.npz", "SEqSI_ZMSE.npz", "AffEq_ZMSE.npz"] # ["ordinary_ZMSE.npz"] # 


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i", "--input")

    args = parser.parse_args()

    input_dir = args.input

    tabs_f = glob(join(input_dir, "*.npz"))

    tabs = np.stack([np.load(f)["results"] for f in tabs_f])
    mean = np.mean(tabs, axis=0)
    std = np.std(tabs, axis=0)


    col_names = [[f"{exp}_{mod}" for mod in met_names] for exp in exp_names]
    col_names = np.concatenate(col_names)

    with_error = []
    n,m = mean.shape
    for i in range(n):
        tmp = []
        for j in range(m):                
            crt_str = f"{round(mean[i,j],3)}" #+'$\\pm$' + f"{round(std[i,j],3)}"
            if mean[i,j]==mean[i].max():
                crt_str = '\\textbf{' + crt_str + "}"
            p = j//4
            if mean[i,j]==mean[i,4*p:4*(p+1)].max():
                crt_str = '\\cellcolor{gray!50}' + crt_str
            
            tmp.append(crt_str)
        with_error.append(np.array(tmp))

    str_tab = np.stack(with_error)
    full_df = pd.DataFrame(data = str_tab, columns= col_names)
    full_df.insert(0, "alter",np.array(alter_names_clean))
    full_df.to_csv(join(input_dir,"mean_tab.csv"), sep='&', index=False)

    print(full_df)

    std_df = pd.DataFrame(data = np.round(std, decimals=3), columns= col_names)
    std_df.insert(0, "alter",np.array(alter_names_clean))
    std_df.to_csv(join(input_dir,"std_tab.csv"), sep='&', index=False)

    print(std_df)

    with_error = []
    n,m = mean.shape
    for i in range(n):
        tmp = []
        for j in range(m):                
            crt_str = f"{round(mean[i,j],3)}" +'$\\pm$' + f"{round(std[i,j],3)}"
            if mean[i,j]==mean[i].max():
                crt_str = '\\textbf{' + crt_str + "}"
            p = j//4
            if mean[i,j]==mean[i,4*p:4*(p+1)].max():
                crt_str = '\\cellcolor{gray!50}' + crt_str
            
            tmp.append(crt_str)
        with_error.append(np.array(tmp))
    
    str_tab = np.stack(with_error)
    print(str_tab)
    split_1_df = pd.DataFrame(data = str_tab[:,:8], columns= col_names[:8])
    split_1_df.insert(0, "alter",np.array(alter_names_clean))
    split_1_df.to_csv(join(input_dir,"split_1.csv"), sep='&', index=False)

    split_2_df = pd.DataFrame(data = str_tab[:,8:], columns= col_names[8:])
    split_2_df.insert(0, "alter",np.array(alter_names_clean))
    split_2_df.to_csv(join(input_dir,"split_2.csv"), sep='&', index=False)