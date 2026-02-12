import torch
from glob import glob
from tifffile import imread, imwrite
from os.path import join, basename, isdir, splitext
from os import mkdir
from scipy.spatial.distance import cdist
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import yaml
import random as rd

from tools.inference_tools import find_centers, pairing
from tools.normalization import norm_a_b
from tools.mask_processing import centers_from_mask

from models.unet import EquivUNet3D
from data.alteration import generate_alteration_parameters, alter_image



corres_method = {"scale-equiv" : "SEq", "norm-equiv" : "AffEq", "ordinary": "ordinary", "seqsi": "SEqSI"}
corres_loss = {"BCE_with_logit" : "BCE", "norm_MSE" : "ZMSE", "min_max_MSE": "ZMSE"}

alterations = ["none", "affine", "scale_low", "scale_high", "shift",
               "affine_saturated", "scale_saturated", "shift_saturated",
               "gamma_light", "gamma_dark", "noise_low", "noise_high"]

TORCH_PRECISION = torch.float64
MAX_DIST = 6
SAVE_EACH_FILE = False


@hydra.main(version_base=None, config_path="configs", config_name="test")
def assess_model(cfg : DictConfig):

    input_dir = cfg["input_dir"]
    models_dir = cfg["models_dir"]
    which_model = cfg["which_model"]
    output_dir = cfg["output_dir"]
    exp_group = cfg["group_idx"]
    tiles = [int(x) for x in cfg["tiles"].split(",")]

    this_model_dir = join(models_dir , f"group_{exp_group}", which_model)
    
    images_dir = join(input_dir, "images")
    masks_dir = join(input_dir, "masks")

    images_path = sorted(glob(join(images_dir, "*.tif")))
    centers_per_img = [centers_from_mask(imread(join(masks_dir,basename(img_f)))) for img_f in images_path]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(join(this_model_dir, "config.yaml"), 'r') as file:
        model_cfg = yaml.safe_load(file)
    mode = model_cfg["experiment"]["equiv"]
    criterion_type = model_cfg["experiment"]["criterion"]
    data_aug_type = model_cfg["data_aug"]["type"]

    SEED = exp_group

    np.random.seed(SEED)
    rd.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    exp_group_dir = join(output_dir, f"group_{exp_group}")
    if not isdir(exp_group_dir): mkdir(exp_group_dir)

    aug_type_dir = join(exp_group_dir, data_aug_type)
    if not isdir(aug_type_dir): mkdir(aug_type_dir)

    assert criterion_type in ["min_max_MSE", "BCE_with_logit", "norm_MSE"]

    method_name = f"{corres_method[mode]}_{corres_loss[criterion_type]}"

    model = EquivUNet3D(mode = mode).to(device=device, dtype=TORCH_PRECISION)
    model.load_state_dict(torch.load(join(this_model_dir, "best.pkl"), map_location=device))

    post_process = lambda x: torch.sigmoid(x.squeeze()) if criterion_type == "BCE_with_logit"\
          else (x.squeeze() - x.mean())/x.std() 


    for alter_type in alterations:
        crt_alter_dir = join(aug_type_dir, alter_type)
        if not isdir(crt_alter_dir): mkdir(crt_alter_dir)
        
        method_dir = join(crt_alter_dir, method_name)
        if not isdir(method_dir): mkdir(method_dir)

        tot_GT = 0
        tot_pred = 0

        all_matchings = list()

        for idx,im_f in enumerate(images_path):
            im_name = basename(im_f)
            print(f"Processing image {im_name}")
            img = imread(im_f)

            normed_img = norm_a_b(img, 0, 1)

            alteration = generate_alteration_parameters(alter_type)
            normed_img = alter_image(normed_img, alteration=alteration)

            normed_img = torch.tensor(normed_img, device=device, dtype=TORCH_PRECISION)[None,None,...]

            with torch.no_grad():
                pred = post_process(model(normed_img)).to(dtype=torch.float32)
            
            th = model_cfg["common_th"]

            print(" # Predicting centers")
            with torch.no_grad():
                pred_centers = find_centers(pred, threshold=th, nb_tiles=tiles).cpu().numpy() 
            print(f" # Nb centers predicted : {len(pred_centers)}")
            print(" # Computing GT centers")
            GT_centers = centers_per_img[idx] 

            if SAVE_EACH_FILE:
                np.savez(join(method_dir, splitext(im_name)[0]+".npz"), pred_centers = pred_centers, GT_centers = GT_centers)
                imwrite(join(method_dir, im_name), pred.cpu().numpy().astype(np.float32))

            tot_GT += len(GT_centers)
            tot_pred += len(pred_centers)
            

            print(" # Matching centers")
            if len(pred_centers)>0:
                C = cdist(pred_centers, GT_centers, metric="euclidean")

                _, _, dist = pairing(C)

                smaller_than_max = (dist < MAX_DIST)

                matching = dist[smaller_than_max]
                
                all_matchings.append(matching)

        matchings_array = np.concatenate(all_matchings)

        hist, _ = np.histogram(matchings_array, bins = [k for k in range(MAX_DIST+1)])
        cum_dist = np.cumsum(hist)
        accuracy_curve = cum_dist/(tot_GT + tot_pred - cum_dist)

        np.savez(join(crt_alter_dir,f"{method_name}.npz"), max_dist = MAX_DIST,  accuracy_curve = accuracy_curve)


if __name__ == "__main__":

    assess_model()
