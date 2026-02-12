import yaml
import torch
from os.path import join
import numpy as np
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.distance import cdist
import hydra

from models.unet import EquivUNet3D
from tools.inference_tools import find_centers, pairing
from tools.mask_processing import centers_from_mask
from tools.optim_tools import iterative_grid_search_min
from data.optimset import OptimSet

@hydra.main(version_base=None, config_path="configs", config_name="optim")
def optim_th(cfg : DictConfig):

    val_path = cfg["set_path"]
    model_dir = cfg["model_dir"]
    which_model = cfg["which_model"]
    model_path = join(model_dir, which_model)

    with open(join(model_path, "config.yaml"), 'r') as file:
        model_cfg = yaml.safe_load(file)

    data_aug_str = model_cfg["data_aug"].get("aug")
    data_aug_list = [] if data_aug_str == None else [x for x in data_aug_str.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode = model_cfg["experiment"]["equiv"]
    criterion_type = model_cfg["experiment"]["criterion"]

    model = EquivUNet3D(mode = mode).to(device=device)

    optim_set = OptimSet(val_path, device=device, data_aug=data_aug_list)
    model.load_state_dict(torch.load(join(model_path,"best.pkl")))
    post_process = lambda x: torch.sigmoid(x.squeeze()) if criterion_type == "BCE_with_logit"\
        else (x.squeeze() - x.mean())/x.std()
    with torch.no_grad():
        infered_val = [post_process(model(optim_set[n][0][None,...])) for n in range(len(optim_set))]

    proba_max = max([infer.max() for infer in infered_val]).item()
    bounds_BCE = proba_max*0.1, proba_max*0.95
    print(bounds_BCE)

    MAX_DIST = 6

    N = len(optim_set)
    GT_centers_list = [centers_from_mask(optim_set[n][1]) for n in range(N)]

    def f_optim(proba_th):

        tot_GT, tot_pred= 0, 0
        pairings_list = list()

        print(f"Call of f for th={proba_th}")

        for n in range(N):
            print(f"\rImg : {n+1}/{N}", end="")
            pred = infered_val[n]

            pred_centers = find_centers(pred, proba_th).cpu().numpy()
            GT_centers = GT_centers_list[n]

            tot_GT += len(GT_centers)
            tot_pred += len(pred_centers)
            if len(pred_centers)>0:
                C = cdist(pred_centers, GT_centers, metric="euclidean")
                _, _, dist = pairing(C)
                smaller_than_max = (dist < MAX_DIST)
                matching = dist[smaller_than_max]
                pairings_list.append(matching)
        
        if tot_pred==0:
            return 0
        
        pairings = np.concatenate(pairings_list)
        hist, _ = np.histogram(pairings, [k for k in range(MAX_DIST+1)])
        cum = np.cumsum(hist)
        curve = cum/(tot_GT+tot_pred-cum)
        score = curve.sum()/MAX_DIST
        print(f"Score : {score}")
        return -score

    new_cfg = {**model_cfg}
    try:
        if (criterion_type == "min_max_MSE" or criterion_type == "norm_MSE"): bounds = (0.5,20) 
        if mode == "ordinary" or mode == "scale-equiv": bounds = bounds_BCE
        opt = iterative_grid_search_min(f_optim, bounds = bounds, verbose = True)
        new_th = opt.item()
        new_cfg = {**new_cfg, "common_th" : new_th}
    except Exception as err:
        print(f"Error : {err}")

    print(yaml.dump(new_cfg))
    OmegaConf.save(config=new_cfg, f=join(model_path, "config.yaml"))

    return new_cfg #OmegaConf.save(config=new_cfg, f=join(model_path, "config.yaml"))


if __name__ == "__main__":

    updated_cfg = optim_th()
