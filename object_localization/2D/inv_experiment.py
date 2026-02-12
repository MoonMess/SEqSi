import torch
from glob import glob
from tifffile import imread
from os.path import join, isdir
from os import mkdir
from scipy.spatial.distance import cdist
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf
import hydra

from tools.inference_tools import find_centers, pairing
from tools.normalization import norm_a_b, norm_lamb_mu

from models.drunet import DRUNet


MAX_DIST = 10

exp_lamb_mu = [
    (1,-2),
    (1,0.5),
    (1,2),
    (1,10),
    (0.5, 0),
    (3,0),
    (255,0),
    (10,-2),
    (0.1,5),
    (3,5)
]

corres_method = {"scale-equiv" : "SEq", "norm-equiv" : "AffEq", "ordinary": "ordinary", "seqsi": "SEqSI"}
corres_loss = {"BCE_with_logit" : "BCE", "norm_MSE" : "ZMSE", "min_max_MSE": "ZMSE"}

PRECISION = torch.float64

@hydra.main(version_base=None, config_path="configs", config_name="inv_exp")
def assess_model_hydra(cfg : DictConfig):
    input_dir = cfg["input_dir"]
    model_dir = cfg["model_dir"]
    which_model = cfg["which_model"]
    output_dir = cfg["output_dir"]
    crt_model = join(model_dir, which_model)
    assess_model(input_dir=input_dir, crt_model=crt_model, output_dir=output_dir)


def assess_model(input_dir, crt_model, output_dir):
    
    images_dir = input_dir

    images = sorted(glob(join(images_dir, "*.tif")))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(join(crt_model, "config.yaml"), 'r') as file:
        model_cfg = yaml.safe_load(file)
    mode = model_cfg["experiment"]["equiv"]
    criterion_type = model_cfg["experiment"]["criterion"]

    method_name = f"{corres_method[mode]}_{corres_loss[criterion_type]}"

    assert criterion_type in ["min_max_MSE", "norm_MSE", "BCE_with_logit"]

    model = DRUNet(blind = True, mode = mode).to(device=device, dtype=PRECISION)
    model.load_state_dict(torch.load(join(crt_model, "best.pkl"), map_location=device))


    ref_centers_list = []

    post_process = lambda x: torch.sigmoid(x.squeeze())  if (criterion_type == "BCE_with_logit")\
        else (x.squeeze() - x.mean())/x.std()
    
    th = model_cfg["common_th"] # model_cfg["basic_th"] if (criterion_type == "BCE_with_logit") else model_cfg["nb_std"]

    print(f"Generating reference centers")
    for idx,im_f in enumerate(images):
        print(f"\r#processing image {idx+1}/{len(images)}", end="")
        img = imread(im_f)

        normed_np = norm_a_b(img, 0, 1)
        normed_img = torch.tensor(normed_np, device=device, dtype=PRECISION)[None,None,...]

        with torch.no_grad():
            pred = post_process(model(normed_img)).cpu()
        
        ref_centers = find_centers(pred, threshold=th).numpy()

        ref_centers_list.append(ref_centers)

    print()
    for param in exp_lamb_mu:

        lamb, mu = param

        exp_name = "shift" if lamb == 1 else ("scale" if mu == 0 else "affine")
        print(f"{exp_name}, lambda = {lamb}, mu = {mu}")
                
        exp_dir = join(output_dir, f"lamb_{lamb}_mu_{mu}")
        if not isdir(exp_dir): mkdir(exp_dir) 

        tot_ref = 0
        tot_new = 0

        all_matchings = list()


        for idx,im_f in enumerate(images):
            print(f"\r#processing image {idx+1}/{len(images)}", end="")
            img = imread(im_f)
            normed_np = norm_lamb_mu(norm_a_b(img, 0, 1), lamb, mu)
            normed_img = torch.tensor(normed_np, device=device, dtype=PRECISION)[None,None,...]

            with torch.no_grad():
                pred = post_process(model(normed_img)).cpu()
        
            new_centers = find_centers(pred, threshold=th).numpy()
            ref_centers = ref_centers_list[idx]

            tot_ref += len(ref_centers)
            tot_new += len(new_centers)

            if len(new_centers)>0:
                C = cdist(new_centers, ref_centers, metric="euclidean")
                _, _, dist = pairing(C)
                smaller_than_max = (dist < MAX_DIST)

                matching = dist[smaller_than_max]
                
                all_matchings.append(matching)
        print()

        if len(all_matchings)>0:
            matchings_array = np.concatenate(all_matchings)
            hist, _ = np.histogram(matchings_array, bins = [k for k in range(MAX_DIST+1)])
            cum_dist = np.cumsum(hist)
            accuracy_curve = cum_dist/(tot_ref + tot_new - cum_dist)
        else:
            accuracy_curve = np.zeros((MAX_DIST))

        np.savez(join(exp_dir,f"{method_name}.npz"), max_dist = MAX_DIST,  accuracy_curve = accuracy_curve, exp_name=exp_name)

        print(f"Invariance : {accuracy_curve[0]}")


if __name__ == "__main__":

    USE_HYDRA = True
    if USE_HYDRA:        
        assess_model_hydra()
    else:
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument("-i","--input")
        parser.add_argument("-o","--output")
        parser.add_argument("-m","--model")

        args = parser.parse_args()

        assess_model(args.input, args.model, args.output)


    
    