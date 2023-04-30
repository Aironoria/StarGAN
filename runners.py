import numpy
import numpy as np

import datasets
from network import MLP
import matplotlib.pyplot as plt
import render
import  torch

from tqdm import tqdm

import metric
def train_epoch(model,optimizer,criterion,dataloader,device="cpu"):
    model.to(device)
    loss_ ,psnr_= metric.AverageMeter(), metric.AverageMeter()
    # iterator = tqdm(dataloader,desc="Train")
    iterator = dataloader
    for item in iterator:
        true_color = item["rgb"].to(device)
        rays_o, rays_d = item["rays_o"].to(device), item["rays_d"].to(device)

        predicted_color = render.render_rays(model,rays_o,rays_d)

        loss = criterion(predicted_color,true_color)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_.update(loss.item())
        psnr_.update(metric.psnr_score(loss.item()))
    return {
        "Loss":loss_.avg,
        "Score":psnr_.avg
    }

def plot_test_image(model,dataset,criterion,device="cpu",save=False,save_path="test.png"):
    model.to(device)
    item = dataset.get_test_image()
    H, W = item["rgb"].shape[1], item["rgb"].shape[2]

    rays_o, rays_d = item["rays_o"].to(device), item["rays_d"].to(device)
    true_color = item["rgb"].reshape(H,W,3)
    predicted_color = []
    with torch.no_grad():
        for i in range(H):
           predicted_color.append(render.render_rays(model, rays_o[:,i,:,:], rays_d[:,i,:,:]).cpu())
    predicted_color = torch.cat(predicted_color,dim=1).reshape(H, W, 3)
    mse = criterion(predicted_color, true_color).item()
    if save:
        plt.imsave(save_path,(predicted_color.numpy() *255).astype(np.uint8))
    else:
        plt.imshow(predicted_color)
        plt.show()
    return{
        "Loss":mse,
        "Score":metric.psnr_score(mse)
    }


def valid_epoch(model,criterion,dataloader,device="cpu"):
    model.to(device)
    loss_, psnr_ = metric.AverageMeter(), metric.AverageMeter()
    # iterator = tqdm(dataloader,desc="Valid")
    iterator = dataloader
    for item in iterator:
        true_color = item["rgb"].to(device)
        rays_o, rays_d = item["rays_o"].to(device), item["rays_d"].to(device)
        with torch.no_grad():
            predicted_color = render.render_rays(model, rays_o, rays_d)
            loss = criterion(predicted_color, true_color)
            loss_.update(loss.item())
            psnr_.update(metric.psnr_score(loss.item()))
    return {
        "Loss": loss_.avg,
        "Score": psnr_.avg
    }

