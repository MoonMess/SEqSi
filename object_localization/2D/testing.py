import torch
from glob import glob
from tifffile import imread, imwrite
from os.path import join, basename, isdir, splitext
from os import mkdir
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import yaml

from tools.inference_tools import find_centers, pairing
from tools.normalization import norm_a_b
from tools.mask_processing import centers_from_mask

from models.drunet import DRUNet


MAX_DIST = 12

@hydra.main(version_base=None, config_path="configs", config_name="test")
def assess_model(cfg : DictConfig):

    bounds = cfg["range"]
    input_dir = cfg["input_dir"]
    model_dir = cfg["model_dir"]
    which_model = cfg["which_model"]
    output_dir = cfg["output_dir"]

    

    range_dir = join(output_dir, bounds)
    if not isdir(range_dir): mkdir(range_dir)
    
    a,b = [float(x) for x in bounds.split(",")]

    images_dir = join(input_dir, "images")
    masks_dir = join(input_dir, "masks")

    images = glob(join(images_dir, "*.tif"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    crt_model = join(model_dir, which_model)

    with open(join(crt_model, "config.yaml"), 'r') as file:
        model_cfg = yaml.safe_load(file)
    mode = model_cfg["experiment"]["equiv"]
    criterion_type = model_cfg["experiment"]["criterion"]

    assert criterion_type in ["min_max_MSE", "norm_MSE", "BCE_with_logit"]

    method_dir = join(range_dir, f"{mode}_{criterion_type}")
    if not isdir(method_dir): mkdir(method_dir)
    print(f"method_dir : {method_dir}")


    model = DRUNet(blind = True, mode = mode).to(device=device)
    model.load_state_dict(torch.load(join(crt_model, "best.pkl"), map_location=device))

    tot_GT = 0
    tot_pred = 0
    tot_match = 0

    all_matchings = list()

    post_process = lambda x: torch.sigmoid(x.squeeze())  if (criterion_type == "BCE_with_logit")\
        else (x.squeeze() - x.mean())/x.std()
    
    th = model_cfg["common_th"] # model_cfg["basic_th"] if (criterion_type == "BCE_with_logit") else model_cfg["nb_std"]

    for im_f in images:
        im_name = basename(im_f)
        #print(f"Processing image {im_name}")
        img = imread(im_f)
        mask = imread(join(masks_dir,im_name))

        normed_np = norm_a_b(img, a, b)
        normed_img = torch.tensor(normed_np, device=device, dtype=torch.float32)[None,None,...]

        with torch.no_grad():
            pred = post_process(model(normed_img)).cpu()
        
        pred_centers = find_centers(pred, threshold=th).numpy()
        #print(f"pred_centers : {pred_centers}")
        GT_centers = centers_from_mask(mask)
        #print(f"GT_centers : {GT_centers}")

        np.savez(join(method_dir, splitext(im_name)[0]+".npz"), pred_centers = pred_centers, GT_centers = GT_centers)
        imwrite(join(method_dir, im_name), pred.detach().numpy())

        tot_GT += len(GT_centers)
        tot_pred += len(pred_centers)
        if len(pred_centers)>0:

            C = cdist(pred_centers, GT_centers, metric="euclidean")

            dist, row_ind, col_ind = pairing(C)

            smaller_than_max = (dist < MAX_DIST)

            matching = dist[smaller_than_max]
            tot_match += len(matching)

            all_matchings.append(matching)

    matchings_array = np.concatenate(all_matchings)

    if len(all_matchings)>0:
        matchings_array = np.concatenate(all_matchings)
        hist, _ = np.histogram(matchings_array, bins = [k for k in range(MAX_DIST+1)])
        cum_dist = np.cumsum(hist)
        accuracy_curve = cum_dist/(tot_GT + tot_pred - cum_dist)
    else:
        accuracy_curve = np.zeros((MAX_DIST))


    print(f"Score = {round(accuracy_curve.sum()/MAX_DIST,3)}")

    np.savez(join(range_dir,f"{mode}_{criterion_type}.npz"), accuracy_curve = accuracy_curve, max_dist = MAX_DIST)


if __name__ == "__main__":

    assess_model()


    
    