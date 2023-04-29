import datasets
from network import MLP
import matplotlib.pyplot as plt
import render
import  torch
import time
def train_epoch(model,optimizer,criterion,dataloader,device="cpu"):
    start = time.time()
    model.to(device)
    total_loss = 0
    for item in dataloader:
        true_color = item["rgb"]
        rays_o, rays_d = item["rays_o"], item["rays_d"]
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        true_color = true_color.to(device)
        predicted_color = render.render_rays(model,rays_o,rays_d)
        loss = criterion(predicted_color,true_color)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # print(loss.item())
    # print( time.time() - start )
    total_loss = total_loss / len(dataloader)
    psnr = -10. * torch.log(loss).item() / torch.log(torch.tensor([10.]))
    print("loss: ",total_loss,"psnr: ",psnr.item())



def valid_epoch(model,optimizer,criterion,dataloader,device="cpu"):
    start = time.time()
    model.to(device)

    for item in dataloader:
        true_color = item["rgb"]
        rays_o, rays_d = item["rays_o"], item["rays_d"]
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)
        true_color = true_color.to(device)
        predicted_color = render.render_rays(model,rays_o,rays_d).cpu().detach().numpy()
        predicted_color= predicted_color.reshape(100, 100, 3)
        plt.imshow(predicted_color)
        plt.show()
    # print( time.time() - start )

