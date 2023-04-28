
import torch
from network import PositionalEncoding


def T_i(i,sigma,delta):
    #sum i points
    return torch.exp(- torch.sum( (sigma * delta)[:,0:i], 1,keepdim=True))


#def pertuerb point:
    # mid = (t[:, :-1] + t[:, 1:]) / 2.
    # lower = torch.cat((t[:, :1], mid), -1)
    # upper = torch.cat((mid, t[:, -1:]), -1)
    # u = torch.rand(t.shape, device=device)
    # t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    # a = (t[:, 1:] - t[:, :-1])

def render_rays(nerf,ray_origins,ray_directions,near=0,far=0.5,bins=192):
    # device = ray_origins.device
    device =torch.device("cpu")
    t= torch.linspace(near,far,bins,device=device).expand(ray_origins.shape[0],bins)
    # x = ray_origins +ray_directions * t
    x =   t.unsqueeze(-1) * ray_directions.unsqueeze(1) + ray_origins.unsqueeze(1)  # [ray_num, samples per ray, 3]
    directions = ray_directions.unsqueeze(1).expand(ray_origins.shape[0],bins,3)

    out = nerf(x,directions)
    out = mlp_raw_to_color(out)


    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(ray_origins.shape[0], 1)), -1)

    breakpoint()
"""
 mlp_raw  = [numbers of rays, samples per ray, 4]
"""
def mlp_raw_to_color(mlp_raw):
    N = mlp_raw.shape[1]
    sigma =mlp_raw[:,:,3]
    delta = torch.ones_like(mlp_raw[:,:,3])  # [numbers of rays, distance_delta]
    Ti = torch.cat([ T_i(i,sigma,delta) for i in range(N) ],-1)
    #ouput = (number of rays, rgb)
    output = torch.cat((
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:, :, 0], -1,keepdim=True),
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:, :, 1], -1,keepdim=True),
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:, :, 2], -1,keepdim=True),
    ),1)
    return output




# mlp_raw_test = torch.rand(3,4,4)
# mlp_raw_test[0,1,-1] = 1
# mlp_raw_test[1,0,-1] = 1
# mlp_raw_test[2,3,-1] = 1
# render(mlp_raw_test)
