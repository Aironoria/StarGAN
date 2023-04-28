from network import MLP
from network import PositionalEncoding
import render
import  torch
def train_epoch(model,optimizer,criterion,dataloader,device="cpu"):
    rays_o, rays_d = torch.rand((90,3)), torch.rand((90,3))
    model = MLP()
    positional_encoder = PositionalEncoding()

    predicted_color = render.render_rays(model,rays_o,rays_d)
    breakpoint()


train_epoch(None,None,None,None)