import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "."))
import torch
import models
from .models import HyperNet
from .models import TargetNet
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
import csv
import os, shutil
def get_block(img_path,size):
    w=size
    h=size
    img_list=[]
    img = Image.open(img_path)
    w0,h0=img.size
    rate=max(w0/4000,h0/4000)
    if w0>4000 or h0>4000:
        img=img.resize((round(w0/rate),round(h0/rate)))
    im=np.array(img)
    for i in range(0,(im.shape[0]//w)+1):
        for j in range(0,(im.shape[1]//h)+1):
            if i==(im.shape[0]//w) and j!=(im.shape[1]//h):
                img_temp=im[-w:,j*h:(j+1)*h,:]
            if j==(im.shape[1]//h) and i!=(im.shape[0]//w):
                img_temp=im[i*w:(i+1)*w,-h:,:]
            if j<(im.shape[1]//h) and i<(im.shape[0]//w):
                img_temp=im[i*w:(i+1)*w,j*h:(j+1)*h,:]
            if i==(im.shape[0]//w) and j==(im.shape[1]//h):
                img_temp=im[-w:,-h:,:]
            img_list.append(img_temp)
    return img_list
def img_transform(img_np_list,size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    new_img_list=[]
    for i in range(len(img_np_list)):
        new_img_list.append(transform(img_np_list[i]).view(1,3,size,size))
    return new_img_list
def get_random_block_one(img_path,size,num=25):
    img=[]
    im = Image.open(img_path)
    w0,h0=im.size
    if w0>h0:
        im = im.resize((512,384))
    else:
        im = im.resize((384, 512))
    im=np.array(im)
    for i in range(0,num):
        h, w = im.shape[:2]
        y = np.random.randint(0, h-size)
        x = np.random.randint(0, w-size)
        im_temp = im[y:y+size, x:x+size, :]
        img.append(im_temp)
    return img
def rename_pics(files_temp,copy_image_file_path,score0):
    count=0
    file_array=os.listdir(files_temp)
    file_array.sort()
    for info in file_array:
        picname=info.split('.j')[-2]
        old_picname=os.path.join(files_temp,info)#
        temp_picname=picname+"_IQA_{}".format(score0[count].round(2))+'.jpg'
        new_picname=os.path.join(copy_image_file_path,temp_picname)
        shutil.copy(old_picname,new_picname)
        count+=1
def single_evaluate(image_path,size):
    img_m=get_random_block_one(image_path,size,num=25)
    img_m=img_transform(img_m,size)
    score_temp=[]
    for index in range(len(img_m)):
        x=img_m[index].cuda(0)
        #img = torch.tensor(x.cuda()).unsqueeze(0)
        paras = model(x)
        model_target = models.TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False
        pred = model_target(paras['target_in_vec'])
        score_temp.append(float(pred.item()))
    score=np.array(score_temp).mean()
    score_std=np.array(score_temp).std()
    return score*0.09, score_std

def single_evaluate_for_system(img,size):
    # img_m=get_random_block_one(image_path,size,num=25)
    img_m=[]
    im = img
    w0,h0=im.size
    if w0>h0:
        im = im.resize((512,384))
    else:
        im = im.resize((384, 512))
    im=np.array(im)
    for i in range(0,25):
        h, w = im.shape[:2]
        y = np.random.randint(0, h-size)
        x = np.random.randint(0, w-size)
        im_temp = im[y:y+size, x:x+size, :]
        img_m.append(im_temp)
    
    img_m=img_transform(img_m,size)
    model = HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    cpuDevice = torch.device('cpu')
    model.eval()
    state_dict = torch.load('models/IQA_Net_v1/koniq_pretrained.pkl',map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model = model.to(cpuDevice)
    score_temp=[]
    for index in range(len(img_m)):
        x=img_m[index].to(cpuDevice)
        #img = torch.tensor(x.cuda()).unsqueeze(0)
        paras = model(x)
        model_target = TargetNet(paras).cuda()
        for param in model_target.parameters():
            param.requires_grad = False
        pred = model_target(paras['target_in_vec'])
        score_temp.append(float(pred.item()))
    score=np.array(score_temp).mean()
    score_std=np.array(score_temp).std()
    return score*0.09, score_std

def multi_evaluate(image_file_path, copy_image_file_path, output_csv):
    img_path_list=[]
    for info1 in os.listdir(image_file_path):
        path2 = os.path.join(image_file_path,info1)
        img_path_list.append(path2)
    img_path_list.sort()
    ID = []
    scores = []
    scores_std = []
    for i in range(len(img_path_list)):
        score, score_std = single_evaluate(img_path_list[i],224)
        ID.append(img_path_list[i])
        scores.append(score)
        scores_std.append(score_std)
    data = pd.DataFrame({"ID": ID, "score": scores, 'score_std': scores_std})
    data.to_csv(output_csv, index=False)
    # 下面这个注释打开，复制所有待测图片，并以得分作为命名
    rename_pics(image_file_path, copy_image_file_path,scores)
    print("Complete!")

if __name__ == '__main__':
    model = models.HyperNet(16, 112, 224, 112, 56, 28, 14, 7)
    model.eval()
    state_dict = torch.load('./koniq_pretrained.pkl',map_location='cuda:0')
    model.load_state_dict(state_dict)
    model = model.to('cuda:0')
    #评测单张图片
    image_path = "./Test_img/D_01.jpg"
    score, score_std = single_evaluate(image_path,224)
    print("score:{}, score_std:{}".format(score, score_std))
    """
    评测多张图片
    输出方式：
    （1）结果写入scv文件中
    （2）复制所有待测图片，并以得分重命名
    输出格式：客观质量评分+评分分布标准差
    """
    # # 结果输出至哪个csv文件
    # output_csv = './result.csv'
    # with open(output_csv, 'w', newline='') as f:
    #     file = csv.writer(f)

    # # 待评测图片所在文件夹
    # image_file_path = './Test_img'

    # # 输出的重命名图片文件夹目录
    # copy_image_file_path = './Test_img_new'

    # if (os.path.exists(copy_image_file_path) == False):
    #     os.makedirs(copy_image_file_path)

    # # 评测一批图片
    # multi_evaluate(image_file_path, copy_image_file_path, output_csv)