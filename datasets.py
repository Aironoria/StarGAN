import json
import os.path
import imageio
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms

class NerfDataset(torch.utils.data.Dataset):
    def __init__(self,data):
        super(NerfDataset,self).__init__()
        self.imgs = data['imgs']
        self.poses = data['poses']
        self.camera_angle_x = data['camera_angle_x']
        self.focal = .5 * self.imgs.shape[2] / np.tan(.5 * self.camera_angle_x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx],self.poses[idx]
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
        poses=[]
        if split =="train" or test_skip==0:
            skip =1
        else:
            skip = test_skip

        for frame in transform["frames"][::skip]:
            file_path = os.path.join(basedir,frame["file_path"]+".png")
            imgs.append(imageio.v2.imread(file_path))
            poses.append(np.array(frame["transform_matrix"]))
        imgs = (np.array(imgs)/225.).astype(np.float32)
        poses = np.array(poses).astype(np.float32)
        data[split] = {"imgs":imgs,
                       "poses":poses,
                       "camera_angle_x":transform["camera_angle_x"]}

        return NerfDataset(data["train"]),NerfDataset(data["val"]), NerfDataset(data["test"])


root ="data/nerf_synthetic"
dataset_name = "lego"
load_data_set(root,dataset_name)

