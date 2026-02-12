import numpy as np
from argparse import ArgumentParser
from os.path import join, basename, splitext
from glob import glob
from os import listdir
import pandas as pd


exp_names = ["none", "Aff", "Naff", "All"]

EXPERIMENT = "augmentation" 
#EXPERIMENT = "artifacts"

if EXPERIMENT == "augmentation":
    alter_names = ["none", "shift", "scale_low", "scale_high", "affine", "shift_saturated", "scale_saturated",
                "affine_saturated", "noise_low", "noise_high", "gamma_dark", "gamma_light"]
    
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
    alter_names = ["arti_1.5", "arti_3", "arti_5", "arti_10"]
    alter_names_clean = [
        "Artifacts (1.5)",
        "Artifacts (3)",
        "Artifacts (5)",
        "Artifacts (10)",
    ]

met_names = ["ordinary_BCE.npz", "SEq_BCE.npz", "SEqSI_ZMSE.npz", "AffEq_ZMSE.npz"] # ["ordinary_ZMSE.npz"] # 


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i", "--input")

    args = parser.parse_args()

    output_dir = args.input
    input_dir = args.input

    full_list = list()

    for exp in exp_names:
        print(f"## Experiment = {exp} ##")

        exp_dir = join(input_dir,exp)
        exp_list = list()
        

        for i,alter in enumerate(alter_names):
            alter_dir = join(exp_dir, alter)

            alter_list = list()

            for met_f in met_names:
                met = join(alter_dir, met_f)

                df = np.load(met)
                accuracy_curve = df["accuracy_curve"]
                max_dist = df["max_dist"]

                score = accuracy_curve.sum()/max_dist

                alter_list.append(round(score.item(),6))
            
            exp_list.append(np.array(alter_list))

        exp_array = np.stack(exp_list)

        full_list.append(exp_array)
        res_df = pd.DataFrame(data = np.array(exp_array), columns=met_names)
        res_df.insert(0, "alter",np.array(alter_names))

        print(res_df)
    
    col_names = [[f"{exp}_{mod}" for mod in met_names] for exp in exp_names]
    col_names = np.concatenate(col_names)
    print(col_names)


    full_tab = np.concatenate(full_list, axis=1)
    tab_path_name = input_dir.split("/")[-2]+".npz"
    tab_path = join(input_dir, tab_path_name)
    np.savez(tab_path, results=full_tab)
    full_df = pd.DataFrame(data = full_tab, columns= col_names)
    full_df.insert(0, "alter",np.array(alter_names_clean))

    full_df.to_csv(join(output_dir,"latex_tab.csv"), sep='&', index=False)

    print(full_df)
                
