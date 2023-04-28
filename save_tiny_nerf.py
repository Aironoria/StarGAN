import os

import numpy as np
import matplotlib.pyplot as plt
import json
data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']
H, W = images.shape[1:3]
camera = 0.6911112070083618
print(images.shape, poses.shape, focal)


os.makedirs("data/nerf_synthetic/tiny_lego")
os.makedirs("data/nerf_synthetic/tiny_lego/train")
os.makedirs("data/nerf_synthetic/tiny_lego/test")
os.makedirs("data/nerf_synthetic/tiny_lego/val")

frames =[]
for i, img in enumerate(images):
    if i <100:
        file_path = f"./train/{str(i).zfill(3)}"
    elif i < 102:
        file_path = f"./val/{str(i).zfill(3)}"
    else:
        file_path = f"./test/{str(i).zfill(3)}"
    frames.append({
        "file_path": file_path,
        "transform_matrix": poses[i].tolist(),
    })

    plt.imsave(os.path.join("data/nerf_synthetic/tiny_lego", file_path+".png"), img)


with open(os.path.join("data/nerf_synthetic/tiny_lego", f"transforms_train.json"), 'w') as file:
    json.dump({ "camera_angle_x": 0.6911112070083618,"frames": frames[:100]},file)

with open(os.path.join("data/nerf_synthetic/tiny_lego", f"transforms_val.json"), 'w') as file:
    json.dump({ "camera_angle_x": 0.6911112070083618,"frames": frames[100:102]},file)

with open(os.path.join("data/nerf_synthetic/tiny_lego", f"transforms_test.json"), 'w') as file:
    json.dump({ "camera_angle_x": 0.6911112070083618,"frames": frames[102:]},file)

testimg, testpose = images[101], poses[101]
images = images[:100,...,:3]
poses = poses[:100]

plt.imshow(testimg)
plt.show()

