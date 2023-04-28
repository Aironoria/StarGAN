import datasets
from network import MLP
from network import PositionalEncoding
import render
import  torch
def train_epoch(model,optimizer,criterion,dataloader,device="cpu"):

    for item in dataloader:
        true_color = item["rgb"]
        rays_o, rays_d = item["rays_o"], item["rays_d"]
        predicted_color = render.render_rays(model,rays_o,rays_d,bins=192)
        loss = criterion(predicted_color,true_color)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())




train_dataset, val_dataset, test_dataset = datasets.load_data_set("data/nerf_synthetic", "tiny_lego")
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)


model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device = torch.device("cpu")
train_epoch(model,optimizer,criterion,train_data_loader,device)

