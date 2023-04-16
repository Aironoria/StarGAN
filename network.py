import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    #deep networks are biased towards learning lower frequency functions,
    #mapping the input to higher dimensional space enables better fitting of data that contains high frequency variation
    def __int__(self,input_c,L):
        super(PositionalEncoding,self).__int__()
        self.input_c= input_c
        self.L = L

    def forward(self,x):
        out =[]
        for i in range(self.L):
            sin_x = torch.sin(torch.pow(2,i)*torch.pi*x)
            cos_x = torch.sin(torch.pow(2,i)*torch.pi*x)
            out.extend([sin_x,cos_x])
        return torch.cat(out,-1)

class MLP(nn.Module):
    def __init__(self,W=256,input_c=60,output_c=3):
        super(MLP,self).__init__()
        self.skip=[4]
        self.layers = nn.ModuleList([
            nn.Linear(input_c,W),
            *[ nn.Linear(W + input_c ,W) if i in self.skips else nn.Linear(W,W)  for i in range(1,8)]
        ])
        self.alpha_linear = nn.Sequential(nn.Linear(W,1), nn.ReLU())
        self.additional_layer=nn.Linear(W,W)
        self.rgb_layer = nn.Sequential( nn.Linear(W+24,W//2),nn.ReLU(inplace=True),
                                          nn.Linear(W//2,3), nn.Sigmoid())

    def forward(self,x,d):
        x_initial =x
        for index,layer in enumerate(self.layers):
            x = F.relu(layer(x))
            if index in self.skip:
                x= torch.cat((x,x_initial),dim=-1)

        alpha = self.alpha_linear(x)
        x = self.additional_layer(x)
        x= torch.cat((x,d),dim=-1)
        rgb = self.rgb_layer(x)
        return torch.cat((rgb,alpha),-1)

