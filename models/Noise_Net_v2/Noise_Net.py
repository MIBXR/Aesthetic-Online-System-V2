import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "."))

import torch
from .mv2 import MobileNetV2
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader
import csv
import math
import os, shutil
def get_block(img, size):
    w=size
    h=size
    img_list=[]
    img = img
    fix_size=2000
    w0,h0=img.size
    rate=max(w0/fix_size,h0/fix_size)
    if w0>fix_size or h0>fix_size:
        img=img.resize((round(w0/rate),round(h0/rate)))
    im=np.array(img)
    for i in range(0,(im.shape[0]//w)+1):
        for j in range(0,(im.shape[1]//h)+1):
            if (im.shape[0]%w)!=0 and i==(im.shape[0]//w) and j!=(im.shape[1]//h):
                img_temp=im[-w:,j*h:(j+1)*h,:]
                ent,mea=get_entropy(img_temp)
                if ent<6 and mea<230 and mea>20: 
                    img_list.append(img_temp)
                continue
            if (im.shape[1]%h)!=0 and j==(im.shape[1]//h) and i!=(im.shape[0]//w):
                img_temp=im[i*w:(i+1)*w,-h:,:]
                ent,mea=get_entropy(img_temp)
                if ent<6 and mea<230 and mea>20: 
                    img_list.append(img_temp)
                continue
            if j<(im.shape[1]//h) and i<(im.shape[0]//w):
                img_temp=im[i*w:(i+1)*w,j*h:(j+1)*h,:]
                ent,mea=get_entropy(img_temp)
                if ent<6 and mea<230 and mea>20: 
                    img_list.append(img_temp)
                continue
            if i==(im.shape[0]//w) and j==(im.shape[1]//h) and ((im.shape[0]%w)!=0 or (im.shape[1]%h)!=0):
                img_temp=im[-w:,-h:,:]
                ent,mea=get_entropy(img_temp)
                if ent<6 and mea<230 and mea>20: 
                    img_list.append(img_temp)
                continue
    if len(img_list)<1:
        img_list=get_random_block_one(img,size,2)
    return img_list
def get_entropy(img):
    img = Image.fromarray(img)
    img = np.array(img.convert('L'))
    tmp = []
    val = 0
    k = 0
    res = 0
    for i in range(256):
        tmp.append(0)
    for i in range(len(img)):
        for j in range(len(img[i])):
            val = img[i][j]
            tmp[val] = float(tmp[val] + 1)
            k =  float(k + 1)
    for i in range(len(tmp)):
        tmp[i] = float(tmp[i] / k)
    for i in range(len(tmp)):
        if(tmp[i] == 0):
            res = res
        else:
            res = float(res - tmp[i] * (math.log(tmp[i]) / math.log(2.0)))
    return res,img.mean()
def img_transform(img_np_list,size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    new_img_list=[]
    for i in range(len(img_np_list)):
        new_img_list.append(transform(img_np_list[i]).view(1,3,size,size))
    return new_img_list
def get_random_block_one(img,size,num_rate):
    img_block=[]
    img = img
    fix_size=2000
    w0,h0=img.size
    rate=max(w0/fix_size,h0/fix_size)
    if w0>fix_size or h0>fix_size:
        img=img.resize((round(w0/rate),round(h0/rate)))
    w,h=img.size
    num=(w//size)*(h//size)//num_rate
    img=np.array(img)
    for i in range(0,num):
        h, w = img.shape[:2]
        y = np.random.randint(0, h-size)
        x = np.random.randint(0, w-size)
        im_temp = img[y:y+size, x:x+size, :]
        img_block.append(im_temp)
    return img_block
def re_score(score,low_flag=0.6):
    if score < low_flag:
        score = 1
    else:
        score = (((score - 0.6) / 0.4) * 8)+1
    return score
def rename_pics(files_temp,copy_image_file_path,score0):
    count=0
    file_array=os.listdir(files_temp)
    file_array.sort()
    for info in file_array:
        picname=info.split('.')[-2]
        old_picname=os.path.join(files_temp,info)#
        temp_picname=picname+"_N_{}".format(score0[count].round(2))+'.jpg'
        new_picname=os.path.join(copy_image_file_path,temp_picname)
        shutil.copy(old_picname,new_picname)
        count+=1
def single_evaluate(image,size):
    model = MobileNetV2()
    model.eval()
    print(current_dir)
    state_dict = torch.load(current_dir + '/pre_train.pth', map_location='cuda:0')
    model.load_state_dict(state_dict)
    model = model.to('cuda:0')
    img_m=get_block(image,size)
    img_m=img_transform(img_m,size)
    score_temp=[]
    for index in range(len(img_m)):
        x=img_m[index].cuda(0)
        score_temp.append(re_score(model(x).item()))
    score=np.array(score_temp).mean()
    # score=re_score(score)
    score_std=np.array(score_temp).std()
    return score, score_std
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
        score, score_std = single_evaluate(img_path_list[i],256)
        ID.append(img_path_list[i])
        scores.append(score)
        scores_std.append(score_std)
    data = pd.DataFrame({"ID": ID, "score": scores, 'score_std': scores_std})
    data.to_csv(output_csv, index=False)
    # 下面这个注释打开，复制所有待测图片，并以得分作为命名
    rename_pics(image_file_path, copy_image_file_path,scores)
    print("Complete!")
if __name__ == '__main__':
    model = MobileNetV2()
    model.eval()
    state_dict = torch.load('pre_train.pth',map_location='cuda:0')
    model.load_state_dict(state_dict)
    model = model.to('cuda:0')
    #评测单张图片
    image_path = "./Test_img/(41)_Realme 6 pro.jpg"
    score, score_std = single_evaluate(image_path,256)
    print("score:{}, score_std:{}".format(score, score_std))
    """
    评测多张图片
    输出方式：
    （1）结果写入scv文件中
    （2）复制所有待测图片，并以得分重命名
    输出格式：噪声评分+评分分布标准差
    """
    # # 结果输出至哪个csv文件
    output_csv = './result.csv'
    with open(output_csv, 'w', newline='') as f:
        file = csv.writer(f)

    # 待评测图片所在文件夹
    image_file_path = './Test_img'

    # 输出的重命名图片文件夹目录
    copy_image_file_path = './Test_img_new'

    if (os.path.exists(copy_image_file_path) == False):
        os.makedirs(copy_image_file_path)

    # 评测一批图片
    multi_evaluate(image_file_path, copy_image_file_path, output_csv)