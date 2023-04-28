
import torch
from network import PositionalEncoding


def T_i(i,sigma,delta):
    #sum i points

    a =(sigma * delta)
    b =  torch.sum( (sigma * delta)[:,0:3], 2,keepdim=True)
    return torch.exp(- torch.sum( (sigma * delta)[:,:,0:i], -1,keepdim=True))


#def pertuerb point:
    # mid = (t[:, :-1] + t[:, 1:]) / 2.
    # lower = torch.cat((t[:, :1], mid), -1)
    # upper = torch.cat((mid, t[:, -1:]), -1)
    # u = torch.rand(t.shape, device=device)
    # t = lower + (upper - lower) * u  # [batch_size, nb_bins]
    # a = (t[:, 1:] - t[:, :-1])

"""
ray_origins = [batch size ,numbers of rays, 3]
"""
def render_rays(nerf,ray_origins,ray_directions,near=0,far=1,bins=10):
    # device = ray_origins.device
    device =torch.device("cpu")
    batch_size = ray_origins.shape[0]
    ray_num = ray_origins.shape[1]
    # expanded_shape  = (ray_origins.shape[0],ray_origins.shape[1])
    t = torch.linspace(near,far,bins,device=device)  #[bins]
    #.expand(*expanded_shape,bins)
    delta = torch.cat((t[1:] - t[ :-1], torch.tensor([1e10], device=device)), -1).expand(batch_size,ray_num ,bins) # [ray_num, samples per ray]

    x = t.expand(batch_size,ray_num,bins).unsqueeze(-1)  * ray_directions.unsqueeze(-2) + ray_origins.unsqueeze(-2)  # [batch_size, ray_num, samples per ray, 3]
    directions = ray_directions.unsqueeze(-2).expand(batch_size,ray_num,bins,3)

    out = nerf(x,directions)

    out = mlp_raw_to_color(out,delta)
    return out
"""
 mlp_raw  = [numbers of rays, samples per ray, 4]
"""
def mlp_raw_to_color(mlp_raw,delta):
    N = mlp_raw.shape[-2]
    sigma =mlp_raw[:,:,:,3]
    # delta = torch.ones_like(mlp_raw[:,:,3])  # [numbers of rays, distance_delta]
    a= T_i(0,sigma,delta)
    Ti = torch.cat([ T_i(i,sigma,delta) for i in range(N) ],-1)

    #ouput = (number of rays, rgb)
    output = torch.cat((
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:,:, :, 0], -1,keepdim=True),
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:,:, :, 1], -1,keepdim=True),
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:, :,:, 2], -1,keepdim=True),
    ),-1)
    return output



#
# mlp_raw_test = torch.rand(1,60,192,4)
#
# mlp_raw_to_color(mlp_raw_test, torch.rand(1,60,192))
