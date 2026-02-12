import numpy as np
from argparse import ArgumentParser
from os.path import join, basename, splitext, isdir
from glob import glob
from os import listdir
import pandas as pd

MAX_DIST = 0

methods = [ "ordinary_BCE.npz", "SEq_BCE.npz",  "AffEq_BCE.npz", "SEqSI_BCE.npz",
         "ordinary_ZMSE.npz", "SEq_ZMSE.npz", "AffEq_ZMSE.npz", "SEqSI_ZMSE.npz"]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i", "--input")

    args = parser.parse_args()

    input_dir = args.input


    exp_list = list()
    met_names = list()
    alter_names = list()

        

    for i,alter in enumerate(sorted(listdir(input_dir))):
        alter_dir = join(input_dir, alter)
        if isdir(alter_dir):

            #methods = sorted(glob(join(alter_dir,"*.npz")))
            alter_names.append(alter)

            alter_list = list()

            for met in methods:
                if i==0: met_names.append(splitext(basename(met))[0])

                df = np.load(join(alter_dir,met))
                accuracy_curve = df["accuracy_curve"]

                alter_list.append(round(accuracy_curve[MAX_DIST].item(),3))
            
            exp_list.append(np.array(alter_list))


    exp_array = np.stack(exp_list)

    I,J = exp_array.shape
    str_list = []
    for i in range(I):
        crt_list = []
        for j in range(J):
            crt_list.append(("\\cellcolor{gray!50}" if exp_array[i,j] == 1 else "")+str(exp_array[i,j]))
        str_list.append(np.array(crt_list))
    #print(exp_array)
    res_df = pd.DataFrame(data = np.array(str_list), columns=met_names)
    res_df.insert(0, "alter",np.array(alter_names))



    res_df.to_csv(join(input_dir,f"latex_tab_d{MAX_DIST}.csv"), sep='&', index=False)

    print(res_df)
                
