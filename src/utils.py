import os  
import sys  
import json  
import pickle  
import random  

import torch  
from torch import nn  
import torch.nn.functional as F  
from tqdm import tqdm  
import matplotlib.pyplot as plt  

# Define the Focal Loss class  
class FocalLoss(nn.Module):  
    def __init__(self, alpha=1, gamma=2, reduce=True):  
        super(FocalLoss, self).__init__()  
        self.alpha = alpha  
        self.gamma = gamma  
        self.reduce = reduce  

    def forward(self, inputs, targets):  
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')  
        pt = torch.exp(-BCE_loss)  
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  
        
        if self.reduce:  
            return torch.mean(F_loss)  
        else:  
            return F_loss  

def read_split_data(root: str, val_rate: float = 0.2):  
    random.seed(0)  # 保证随机结果可复现  
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)  

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]  
    flower_class.sort()  
    class_indices = dict((k, v) for v, k in enumerate(flower_class))  
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)  
    with open('class_indices.json', 'w') as json_file:  
        json_file.write(json_str)  

    train_images_path = []  
    train_images_label = []  
    val_images_path = []  
    val_images_label = []  
    every_class_num = []  
    supported = [".jpg", ".JPG", ".png", ".PNG"]  

    for cla in flower_class:  
        cla_path = os.path.join(root, cla)  
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)  
                  if os.path.splitext(i)[-1] in supported]  
        images.sort()  
        image_class = class_indices[cla]  
        every_class_num.append(len(images))  
        val_path = random.sample(images, k=int(len(images) * val_rate))  

        for img_path in images:  
            if img_path in val_path:  
                val_images_path.append(img_path)  
                val_images_label.append(image_class)  
            else:  
                train_images_path.append(img_path)  
                train_images_label.append(image_class)  

    print("{} images were found in the dataset.".format(sum(every_class_num)))  
    print("{} images for training.".format(len(train_images_path)))  
    print("{} images for validation.".format(len(val_images_path)))  
    assert len(train_images_path) > 0, "number of training images must greater than 0."  
    assert len(val_images_path) > 0, "number of validation images must greater than 0."  

    plot_image = False  
    if plot_image:  
        plt.bar(range(len(flower_class)), every_class_num, align='center')  
        plt.xticks(range(len(flower_class)), flower_class)  
        for i, v in enumerate(every_class_num):  
            plt.text(x=i, y=v + 5, s=str(v), ha='center')  
        plt.xlabel('image class')  
        plt.ylabel('number of images')  
        plt.title('flower class distribution')  
        plt.show()  

    return train_images_path, train_images_label, val_images_path, val_images_label  

def plot_data_loader_image(data_loader):  
    batch_size = data_loader.batch_size  
    plot_num = min(batch_size, 4)  

    json_path = './class_indices.json'  
    assert os.path.exists(json_path), json_path + " does not exist."  
    json_file = open(json_path, 'r')  
    class_indices = json.load(json_file)  

    for data in data_loader:  
        images, labels = data  
        for i in range(plot_num):  
            img = images[i].numpy().transpose(1, 2, 0)  
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255  
            label = labels[i].item()  
            plt.subplot(1, plot_num, i+1)  
            plt.xlabel(class_indices[str(label)])  
            plt.xticks([])  
            plt.yticks([])  
            plt.imshow(img.astype('uint8'))  
        plt.show()  

def write_pickle(list_info: list, file_name: str):  
    with open(file_name, 'wb') as f:  
        pickle.dump(list_info, f)  

def read_pickle(file_name: str) -> list:  
    with open(file_name, 'rb') as f:  
        info_list = pickle.load(f)  
        return info_list  

def train_one_epoch(model, optimizer, data_loader, device, epoch, criterion):  
    model.train()  
    accu_loss = torch.zeros(1).to(device)  
    accu_num = torch.zeros(1).to(device)  
    optimizer.zero_grad()  

    sample_num = 0  
    data_loader = tqdm(data_loader, file=sys.stdout)  
    for step, data in enumerate(data_loader):  
        images, labels = data  
        sample_num += images.shape[0]  

        pred = model(images.to(device))  
        pred_classes = torch.max(pred, dim=1)[1]  
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()  

        loss = criterion(pred, labels.to(device))  # Use Focal Loss  
        loss.backward()  
        accu_loss += loss.detach()  

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,  
                                                                               accu_loss.item() / (step + 1),  
                                                                               accu_num.item() / sample_num)  

        if not torch.isfinite(loss):  
            print('WARNING: non-finite loss, ending training ', loss)  
            sys.exit(1)  

        optimizer.step()  
        optimizer.zero_grad()  

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num  

@torch.no_grad()  
def evaluate(model, data_loader, device, epoch, criterion):  
    model.eval()  

    accu_num = torch.zeros(1).to(device)  
    accu_loss = torch.zeros(1).to(device)  

    sample_num = 0  
    data_loader = tqdm(data_loader, file=sys.stdout)  
    for step, data in enumerate(data_loader):  
        images, labels = data  
        sample_num += images.shape[0]  

        pred = model(images.to(device))  
        pred_classes = torch.max(pred, dim=1)[1]  
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()  

        loss = criterion(pred, labels.to(device))  # Use Focal Loss  
        accu_loss += loss  

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,  
                                                                               accu_loss.item() / (step + 1),  
                                                                               accu_num.item() / sample_num)  

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num