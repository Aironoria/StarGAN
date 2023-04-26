import numpy as np
import torch


def get_rays_directions(H,W,K):
    pass
def get_rays(H,W,K,c2w):
    j,i = torch.meshgrid(torch.arange(H),torch.arange(W))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    directions = torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1) # (H, W, 3)
    rays_d =directions @ c2w[:3,:3]
    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)  # (H, W, 3)
    rays_o = torch.tensor(c2w[:3, -1]).expand(rays_d.shape)  # (H, W, 3)
    return rays_o, rays_d

