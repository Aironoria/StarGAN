import torch
import datasets
from network import MLP
import matplotlib.pyplot as plt
import runners
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
ray_num =1024

save_root = os.path.join("res","04_30_tiny_lego")
test_plot_dir = os.path.join(save_root,"test_plot")
os.makedirs(test_plot_dir,exist_ok=True)
writer = SummaryWriter(os.path.join(save_root,"log"))

if not os.path.exists(test_plot_dir):
    os.makedirs(test_plot_dir)


train_dataset, val_dataset, test_dataset = datasets.load_data_set("data/nerf_synthetic", "tiny_lego",ray_num=ray_num)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

model = MLP()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

with tqdm( range(10000) ) as t:
    max_score =  0
    for i in t:
        train_log = runners.train_epoch(model,optimizer,criterion,train_data_loader,device=device)
        writer.add_scalar("train/loss",train_log["Loss"],i)
        writer.add_scalar("train/score",train_log["Score"],i)
        if i % 10 == 0:
            valid_log = runners.valid_epoch(model,criterion,val_data_loader,device=device)
            if valid_log["Score"] > max_score:
                max_score = valid_log["Score"]
                torch.save(model.state_dict(), os.path.join(save_root,"model.pth"))
            writer.add_scalar("valid/loss",valid_log["Loss"],i)
            writer.add_scalar("valid/score",valid_log["Score"],i)

        if ( i < 200 and i %10 ==0) or i % 100 == 0:
            runners.plot_test_image(model,test_dataset,criterion,device=device,save=True,save_path=os.path.join(test_plot_dir,str(i).zfill(5)+".png"))
