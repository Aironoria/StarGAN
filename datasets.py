import json
import os.path
import imageio
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms

from rayUtils import get_rays


class NerfDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        super(NerfDataset,self).__init__()
        self.imgs = data['imgs']
        self.c2ws = data['c2ws']
        self.camera_angle_x = data['camera_angle_x']
        self.read_meta()


    def read_meta(self):
        self.H = self.imgs.shape[1]
        self.W = self.imgs.shape[2]
        self.focal = .5 * self.W / np.tan(.5 * self.camera_angle_x)
        self.K = np.array([
            [self.focal, 0, 0.5 * self.W],
            [0, self.focal, 0.5 * self.H],
            [0, 0, 1]
        ])
        self.near = 2
        self.far  = 6
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self,idx):
        img = self.imgs[idx]  # (H, W, 4)
        # img = np.array(Image.open(self.img_paths[idx])).astype(np.float32) / 255.
        c2w = self.c2ws[idx]
        rays_o, rays_d = get_rays(self.H, self.W, self.K, c2w)
        # h=300
        # w=400
        # img = img[h:w,h:w, :]
        # rays_o = rays_o[h:w,h:w, :]
        # rays_d = rays_d[h:w,h:w, :]
        # rays = torch.cat([rays_o, rays_d, self.near * torch.ones_like(rays_o[:, :, :1]),
        #                     self.far * torch.ones_like(rays_o[:, :, :1])], -1) # (H, W, 8)

        #Color = Color * alpha + Background * (1 - alpha);
        # rgb = img[..., :3] * img[..., -1:] + 0* (1. - img[..., -1:])
        rgb = img[..., :3]
        rgb = rgb.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        partion_size = 400*400
        sample = {
            "rgb": rgb,
            "rays_o": rays_o,
            "rays_d": rays_d,
            "c2w": c2w,
        }
        return sample
# lego: blender
def load_data_set(root,dataset_name,image_size=None):
    test_skip=0
    splits = ["train","val","test"]
    basedir = os.path.join(root,dataset_name)
    transforms ={}
    for dir in splits:
        with open(os.path.join(basedir, f"transforms_{dir}.json"),'r') as file:
            transforms[dir] = json.load(file)
    data={}
    for split in splits:
        transform = transforms[split]
        imgs =[]
        c2ws=[]
        if split =="train" or test_skip==0:
            skip =1
        else:
            skip = test_skip

        for frame in transform["frames"][::skip]:
            file_path = os.path.join(basedir,frame["file_path"]+".png")
            imgs.append(imageio.v2.imread(file_path))
            c2ws.append(np.array(frame["transform_matrix"]))
        # a = (np.array(imgs)/225.).astype(np.float32)
        imgs = np.array(imgs).astype(np.float32)/255.
        c2ws = np.array(c2ws).astype(np.float32)
        data[split] = {"imgs":imgs,
                       "c2ws":c2ws,
                       "camera_angle_x":transform["camera_angle_x"]}

    return NerfDataset(data["train"]),NerfDataset(data["val"]), NerfDataset(data["test"])


if __name__ == '__main__':
    root = "data/nerf_synthetic"
    dataset_name = "lego"
    load_data_set(root, dataset_name)[0][0]

