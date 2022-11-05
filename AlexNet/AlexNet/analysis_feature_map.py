import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
# from AlexNet import AlexNet
from AlexNet import AlexNet
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

data_transform = transforms.Compose(
    [transforms.Resize((224,224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
# create model
model = AlexNet(num_classes=5)
# load model weights
model_weight_path = "E:\\系统默认\\桌面\\AlexNet\\AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

# load image
image = Image.open("E:\\系统默认\\桌面\\AlexNet\\AlexNet\\1.jpg")
# [N,C,H,W]
img = data_transform(image)
# expand batch dimension
img = torch.unsqueeze(img,dim=0)

# forward
out_put = model(img)

for feature_map in out_put:
    #[N,C,H,W]->[C,H,W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C,H,W]->[H,W,C]
    im = np.transpose(im,[1,2,0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3,4,i+1)
        # [H,W,C]
        plt.imshow(im[:,:,i],cmap='gray')
    plt.show()
