import torch
import torch.utils
import torch.utils.data
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from os.path import join, isdir
from os import mkdir
from datetime import datetime
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import random as rd

from models.unet import EquivUNet3D 
from data.equiset import EquiSet
from data.optimset import OptimSet
from loss.norm_mse import NormalizedMSE
from loss.norm_bce import NormalizedBCE
from tools.inference_tools import find_centers, pairing
from tools.mask_processing import centers_from_mask
from tools.plot_tools import plot_depths_map
from tools.normalization import norm_a_b
from tools.optim_tools import iterative_grid_search_min

mode_dict = {"norm-equiv": "AffEq", "seqsi" : "SEqSI", "scale-equiv" : "SEq", "ordinary" : "ordinary"}


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg : DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    
    batch_size = cfg["parameters"]["batch_size"]
    nb_epochs = cfg["parameters"]["nb_epochs"]
    lr = cfg["parameters"]["lr"]

    group = cfg["exp_group"]
    SEED = group
    np.random.seed(SEED)
    rd.seed(SEED)
    torch.manual_seed(SEED)

    mode = cfg["experiment"]["equiv"]
    assert mode in ["ordinary", "norm-equiv", "seqsi", "scale-equiv" ]

    criterion_type = cfg["experiment"]["criterion"]
    assert criterion_type in  ["norm_MSE", "BCE_with_logit", "norm_BCE"]

    data_aug_cfg = cfg["data_aug"]
    data_aug_type = "" if data_aug_cfg["type"]=="none" else f"-{data_aug_cfg['type']}"
    data_aug_str = data_aug_cfg.get("aug")
    data_aug_list = [] if data_aug_str == None else ([x for x in data_aug_str.split(",")] + ["none"])

    patch_size = cfg["data"]["patch_size"]
    train_path = cfg["data"]["train_set"]
    val_path = cfg["data"]["val_set"]
    
    loss_norm_type = cfg["experiment"].get("norm_type")

    tag = cfg["tag"]

    save_path = cfg["save"]["path"]

    now = datetime.now()
    model_name = f"{mode_dict[mode]}-{criterion_type}{data_aug_type}_{str(now).replace(' ','_').split('.')[0]}"

    experiment_group_path = join(save_path, f"group_{group}")
    if not isdir(experiment_group_path): mkdir(experiment_group_path)
    model_path = join(experiment_group_path, model_name)
    if not isdir(model_path): mkdir(model_path)

    model = EquivUNet3D(mode=mode).to(device) 

    if criterion_type == "norm_MSE":
        criterion = NormalizedMSE(norm_type=loss_norm_type) 
    if criterion_type == "BCE_with_logit":
        criterion = torch.nn.BCEWithLogitsLoss()

    trainset = EquiSet(path = train_path, patch_size = patch_size, device = device, data_aug=data_aug_list)
    valset = EquiSet(path = val_path, patch_size = patch_size, device = device, data_aug=data_aug_list)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    min_val_loss = float("inf")

    wandb.init(project="norm-eq-center-3D", name=f"{model_name}_{tag}", config=dict(cfg))

    for ep in range(nb_epochs):

        print(f"Training epoch : {ep}")

        N = len(train_loader)
        M = len(val_loader)
        tot_loss = 0
        val_loss = 0

        max_grad_ep = 0

        for batch in train_loader:
            optim.zero_grad()

            imgs, proba = batch

            pred = model(imgs)

            loss = criterion(pred[:,0], proba)

            loss.backward()

            max_grad = max([x.grad.abs().max().item() for x in model.parameters()])
            print(f"Max gradient :{max_grad}")
            if max_grad>max_grad_ep:
                max_grad_ep=max_grad

            # Apply gradient clipping
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

            optim.step()

            tot_loss += loss.item()
        
        
        with torch.no_grad():
            for batch in val_loader:

                imgs, proba = batch
                pred = model(imgs)

                loss = criterion(pred[:,0], proba)

                mean_pred = pred.abs().mean().item()

                val_loss += loss.item()

        print(f"Train loss : {round(tot_loss/N, 4)} - Val loss : {round(val_loss/M, 4)}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

        
        pred_plot = torch.sigmoid(pred[0,0].detach()).cpu().numpy() if criterion_type == "BCE_with_logit"\
            else norm_a_b(pred[0,0].detach().cpu().numpy(), 0, 1) 
    
        
        plot_depths_map(proba[0].detach().cpu().numpy(), ax1)
        plot_depths_map(pred_plot, ax2)

        wandb.log({"train_loss" : tot_loss/N, "val_loss" : val_loss/M, "plots" : fig,
                   "mean_pred" : mean_pred, "max_grad" : max_grad_ep})

        plt.close(fig)
        

        crt_val_loss = val_loss/M
        if crt_val_loss < min_val_loss:
            torch.save(model.state_dict(), join(model_path, "best.pkl"))
            min_val_loss = crt_val_loss


    wandb.finish()

    optim_set = OptimSet(val_path, device=device, data_aug=data_aug_list)
    model.load_state_dict(torch.load(join(model_path, "best.pkl")))
    post_process = lambda x: torch.sigmoid(x.squeeze()) if criterion_type == "BCE_with_logit"\
        else (x.squeeze()-x.mean())/x.std()
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

    new_cfg = {**cfg}

    try:
        if (criterion_type == "min_max_MSE" or criterion_type == "norm_MSE"): bounds = (0.5,20) 
        if mode == "ordinary" or mode == "scale-equiv": bounds = bounds_BCE
        opt = iterative_grid_search_min(f_optim, bounds = bounds, verbose = True)
        new_th = opt.item()
        new_cfg = {**new_cfg, "common_th" : new_th}
    except Exception as err:
        print(f"Error : {err}")


    OmegaConf.save(config=new_cfg, f=join(model_path, "config.yaml"))


if __name__ == "__main__":
    train()