######################################## 训练脚本编写第一步：导入需要的包 ##########################################################################################
import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from AlexNet import AlexNet

############################################# 训练脚本主函数定义 ###################################################################################################
def main():

    ############# 指定训练设备 #####################################################################################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    ############# 数据预处理定义 ####################################################################################################################################
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        ######### val不需要任何数据增强 ####################################################################################################################################
        "val": transforms.Compose([transforms.Resize((224, 224)),  
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    ################# 指定训练数据集的路径，注意数据集路径, 指定到flower这一目录级别  ####################################################################################
    data_path = os.path.abspath(os.path.join(os.getcwd(), "E:\\系统默认\\桌面\\AlexNet\\AlexNet\\flower"))   
    
    ################# 判断训练数据集的路径是否存在，如果不存在则打印错误提示 #############################################################################################
    if os.path.exists(data_path) is False:
        print('dataset path error')
    
    ################# 判断训练数据集的路径是否存在，如果不存在则打印错误提示，跟上一句的作用一样，随便写那一句都行，两句都写上也不影响程序的运行 ################################
    assert os.path.exists(data_path), "{} path does not exist.".format(data_path)
  
    ################# 指定训练数据集的路径，随便写那种都行，都写上也不影响程序的运行 #######################################################################################
    data_path='E:\\系统默认\\桌面\\AlexNet\\AlexNet\\flower'

    ################# 使用ImageFlolder加载数据集中的图像，并使用指定的预处理处理图像， ImageFlolder会同时返回图像和对应的标签。##############################################
    train_dataset = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(data_path, "val"), transform=data_transform["val"])
    
    ############ 计算训练数据集和验证数据集图片的数量 ####################################################################################################################
    train_num = len(train_dataset)
    val_num = len(validate_dataset)

    ################ 使用class_to_idx给类别一个index，作为训练时的标签： {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4} ################################
    flower_list = train_dataset.class_to_idx

    ################ 创建一个字典，存储index和类别的对应关系，在模型推理阶段会用到。{'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4} ########################
    cla_dict = dict((val, key) for key, val in flower_list.items())

    ################ 将字典写成一个json文件 ##############################################################################################################################
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    ###### batch_size大小，是超参，可调，如果模型跑不起来，尝试调小batch_size ###############################################################################################
    batch_size = 32 

    ########################## 用于加载数据集的进程数量 ###################################################################################################################
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  
    print('Using {} dataloader workers every process'.format(nw))

    ########### 使用 DataLoader 将 ImageFloder 加载的数据集处理成批量（batch）加载模型 ######################################################################################
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)
    
    ######################### 注意，验证集不需要shuffle ###################################################################################################################
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4, shuffle=False,  num_workers=nw) 
    print("using {} images for training, {} images for validation.".format(train_num, val_num))
    
    ######################################### 实例化模型，并送进设备 #######################################################################################################
    net = AlexNet(num_classes=5, init_weights=True)
    net.to(device)

    ##################### 指定损失函数用于计算损失 #########################################################################################################################
    loss_function = nn.CrossEntropyLoss() 

    ##################### 指定优化器用于更新模型参数 #######################################################################################################################
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    ######################## 指定训练迭代的轮数 #############################################################################################################################
    epochs = 20

    ################## 指定训练权重的存储地址 ################################################################################################################################
    save_path = './AlexNet.pth'

    best_acc = 0.0

    ################### train_stepa是数据集可以被分成的批次数量 = num(dataset) / batch_size ####################################################################################
    train_steps = len(train_loader) 

    ################## epoch是迭代训练数据集的次数 #############################################################################################################################
    for epoch in range(epochs):

        ################################################## train #############################################################################################################
        net.train()
        running_loss = 0.0

        ############################ tqdm是一个进度条显示器，可以在终端打印出现在的训练进度 #######################################################################################
        train_bar = tqdm(train_loader, file=sys.stdout, ncols=100)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))

            ####### 求损失 #####################################################################################################################################################
            loss = loss_function(outputs, labels.to(device)) 

            ############## 自动求导 ############################################################################################################################################
            loss.backward() 

            ############## 梯度下降 #############################################################################################################################################
            optimizer.step() 

            ############################# print statistics #####################################################################################################################
            running_loss += loss.item()

            ########################### .desc是进度条tqdm中的成员变量，作用是描述信息 ##############################################################################################
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,  epochs, loss)

        ############################################################################## validate #################################################################################
        net.eval()

        ################## accumulate accurate number / epoch ###################################################################################################################
        acc = 0.0  
        
        with torch.no_grad(): 
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item() 

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %  (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
