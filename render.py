
import torch



def T_i(i,sigma,delta):
    #sum i points
    return torch.exp(- torch.sum( (sigma * delta)[:,0:i], 1,keepdim=True))

"""
 mlp_raw  = [numbers of rays, samples per ray, 4]
"""
def render(mlp_raw):
    N = mlp_raw.shape[1]
    sigma =mlp_raw[:,:,3]
    delta = torch.ones_like(mlp_raw[:,:,3])  # [numbers of rays, distance_delta]
    Ti = torch.cat([ T_i(i,sigma,delta) for i in range(N) ],-1)
    output = torch.cat((
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:, :, 0], -1,keepdim=True),
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:, :, 1], -1,keepdim=True),
        torch.sum(Ti * (1 - torch.exp(- sigma * delta)) * mlp_raw[:, :, 2], -1,keepdim=True),
    ),1)
    c_r = mlp_raw[:,:,0].numpy()
    Ti =Ti.numpy()
    output = output.numpy()
    sigma = sigma.numpy()

    breakpoint()


mlp_raw_test = torch.rand(3,4,4)
mlp_raw_test[0,1,-1] = 1
mlp_raw_test[1,0,-1] = 1
mlp_raw_test[2,3,-1] = 1
render(mlp_raw_test)