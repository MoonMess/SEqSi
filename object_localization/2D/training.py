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
from scipy.optimize import linear_sum_assignment, minimize_scalar
import numpy as np
import random as rd

from models.drunet import DRUNet as UNet
from data.equiset import EquiSet
from data.optimset import OptimSet
from loss.normalized_mse import NormalizedMSE
from tools.inference_tools import find_centers, pairing
from tools.mask_processing import centers_from_mask


@hydra.main(version_base=None, config_path="configs", config_name="train")
def train(cfg : DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    blind = True
    
    batch_size = cfg["parameters"]["batch_size"]
    nb_epochs = cfg["parameters"]["nb_epochs"]
    lr = cfg["parameters"]["lr"]

    mode = cfg["experiment"]["equiv"]
    assert mode in ["ordinary", "norm-equiv", "seqsi", "scale-equiv" ]

    criterion_type = cfg["experiment"]["criterion"]
    assert criterion_type in  ["norm_MSE", "BCE_with_logit"]

    patch_size = cfg["data"]["patch_size"]
    train_path = cfg["data"]["train_set"]
    val_path = cfg["data"]["val_set"]

    save_path = cfg["save"]["path"]

    if criterion_type == "norm_MSE":
        norm_type = cfg["experiment"]["norm_type"]

    group = cfg["seed"]
    SEED = group
    np.random.seed(SEED)
    rd.seed(SEED)
    torch.manual_seed(SEED)

    now = datetime.now()
    model_name = f"centroid-{mode}-{criterion_type}_{str(now).replace(' ','_').split('.')[0]}"

    seed_dir = join(save_path, f"seed_{group}")
    if not isdir(seed_dir): mkdir(seed_dir)

    model_path = join(seed_dir, model_name)
    if not isdir(model_path): mkdir(model_path)

    model = UNet(blind=blind, mode=mode).to(device) 

    if criterion_type == "norm_MSE":
        criterion = NormalizedMSE(norm_type=norm_type)
    if criterion_type == "BCE_with_logit":
        criterion = torch.nn.BCEWithLogitsLoss()

    trainset = EquiSet(path = train_path, patch_size = patch_size, device = device)
    valset = EquiSet(path = val_path, patch_size = patch_size, device = device)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    min_val_loss = float("inf")

    wandb.init(project="norm-equiv-centroids", name=model_name, config=dict(cfg))

    OmegaConf.save(config=cfg, f=join(model_path, "config.yaml"))


    ### Training

    for ep in range(nb_epochs):

        print(f"Training epoch : {ep}")

        N = len(train_loader)
        M = len(val_loader)
        tot_loss = 0
        val_loss = 0

        for batch in train_loader:
            optim.zero_grad()

            imgs, proba = batch

            pred = model(imgs)

            loss = criterion(pred[:,0], proba)

            loss.backward()
            optim.step()

            tot_loss += loss.item()
        
        
        with torch.no_grad():
            for batch in val_loader:

                imgs, proba = batch
                pred = model(imgs)

                loss = criterion(pred[:,0], proba)

                val_loss += loss.item()
        
        pred_plot = wandb.Image(pred[0,0].detach().cpu().numpy()) if criterion_type == "norm_MSE"\
            else  wandb.Image(torch.sigmoid(pred[0,0].detach()).cpu().numpy())

        wandb.log({"train_loss" : tot_loss/N, "val_loss" : val_loss/M, "val_img" : wandb.Image(imgs[0,0].detach().cpu().numpy()),
                   "val_pred" : pred_plot, "val_GT" : wandb.Image(proba[0].detach().cpu().numpy())})
        

        crt_val_loss = val_loss/M
        if crt_val_loss < min_val_loss:
            min_val_loss = crt_val_loss
            torch.save(model.state_dict(), join(model_path, "best.pkl"))


    wandb.finish()


    ### Thresholding optimization
    
    optim_set = OptimSet(val_path, device=device)
    model.load_state_dict(torch.load(join(model_path, "best.pkl")))
    post_process = lambda x: torch.sigmoid(x.squeeze())  if (criterion_type == "BCE_with_logit")\
        else (x.squeeze() - x.mean())/x.std()
    with torch.no_grad():
        infered_val = [post_process(model(optim_set[n][0][None,...])) for n in range(len(optim_set))]

    GT_centers_list = [centers_from_mask(optim_set[n][1]) for n in range(len(optim_set))]

    proba_max = max([infer.max() for infer in infered_val]).item()
    bounds_BCE = proba_max*0.1, proba_max*0.95
    print(bounds_BCE)

    MAX_DIST = 12
    
    def f_generic(th):
        N = len(optim_set)

        tot_GT, tot_pred = 0, 0
        all_matchings = list()

        for n in range(N):
            pred = infered_val[n]

            proba_th = th
            pred_centers = find_centers(pred, proba_th).cpu().numpy()
            GT_centers = GT_centers_list[n] # centers_from_mask(mask)

            tot_GT += len(GT_centers)
            tot_pred += len(pred_centers)
            if len(pred_centers)>0:
                C = cdist(pred_centers, GT_centers, metric="euclidean")
                _, _, dist = pairing(C)
                smaller_than_max = (dist < MAX_DIST)
                matching = dist[smaller_than_max]
                all_matchings.append(matching)

        if len(all_matchings)>0:
            matchings_array = np.concatenate(all_matchings)
            hist, _ = np.histogram(matchings_array, bins = [k for k in range(MAX_DIST+1)])
            cum_dist = np.cumsum(hist)
            accuracy_curve = cum_dist/(tot_GT + tot_pred - cum_dist)
        else:
            accuracy_curve = np.zeros((MAX_DIST))
        
        score = accuracy_curve.sum()/MAX_DIST
        print(f"crt_th = {round(th,3)} ; score = {round(score,3)}")

        return -score

    new_cfg = {**cfg}
    bounds = [0.5, 5.] if criterion_type == "norm_MSE" else bounds_BCE
    opt = minimize_scalar(f_generic, method="bounded", bounds = bounds, options={"maxiter" : 10})
    new_cfg = {**new_cfg, "common_th" :opt.x.item()}

    OmegaConf.save(config=new_cfg, f=join(model_path, "config.yaml"))



if __name__ == "__main__":
    train()