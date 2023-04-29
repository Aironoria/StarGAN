import numpy as np
import torch
from tqdm import tqdm as tdqm
import rayUtils
import render
from network import MLP
import imageio
trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=float)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1],
], dtype=float)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1],
], dtype=float)


def get_c2w(theta,phi,radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) @ c2w
    return torch.tensor(c2w).float()

frames =[]
H=W=100
focal = 138.88887889922103
K =  np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])
model = MLP()
model.load_state_dict(torch.load("model.pth"))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
for th in tdqm(np.linspace(0,360,120)):
    with torch.no_grad():
        c2w = get_c2w(th, -30, 4)
        rays_o, rays_d = rayUtils.get_rays(H, W, K, c2w)
        rays_o, rays_d = rays_o.to(device), rays_d.to(device)
        predicted_color = render.render_rays(model, rays_o, rays_d).reshape(H, W, 3).cpu().numpy()
        frames.append((255 * predicted_color).astype(np.uint8))
imageio.mimsave('video.gif', frames, duration=1000/30)
