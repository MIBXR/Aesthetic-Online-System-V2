# coding:utf-8
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "."))

import torch.nn as nn
from models.mv2 import mobile_net_v2
import torch
import os, shutil
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import numpy as np
from models.color_hue import image_colorfulness
import cv2
import models.color_harmonization as color_harmonization
from models.wheel import orig_score

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
    mean=IMAGE_NET_MEAN,
    std=IMAGE_NET_STD)

def TransformPicture(x):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize])
    return transform(x)

def get_score(y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    # w = w.to('cuda')
    w_batch = w.repeat(y_pred.size(0), 1)
    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

def re_score(score, low_flag=1.8, high_flag=8.6):
    if score < 5:
        if score <= low_flag:
            score = 0.1
        else:
            score = (score - low_flag) / (5 - low_flag) * 5
    else:
        if score >= high_flag:
            score = 9.8
        else:
            score = ((score - 5) / (high_flag - 5) * 5) + 5
    return score

def enhance_score(net_score, hue_score):
    if hue_score == 0:
        return 0
    if hue_score < 50:
        color_score = net_score - pow(2,((50 - hue_score) / 50)) 
    else:
        color_score = net_score + pow(2,((hue_score - 50) / 50)) 
    # high_probability = high_probability/100
    # if high_probability > 0.95 and net_score > 5:
    #     color_score += high_probability
    # elif high_probability > 0.95 and net_score < 5:
    #     color_score += high_probability / 2
    # elif high_probability < 0.7 and net_score > 5:
    #     color_score -= (1 - high_probability)
    # elif high_probability < 0.7 and net_score < 5:
    #     color_score -= (1 - high_probability) * 2
    if color_score < 0:
        color_score = 0
    if color_score > 10:
        color_score = 10
    # print("net",net_score,"color",color_score)
    return color_score

def getMessage(image,color_score):
    message1 = ''
    message2 = ''
    if color_score < 3:
        message1 = '比较差,'
    elif color_score >= 3 and color_score < 5:
        message1 = '比较一般,'
    elif color_score >= 5 and color_score < 7:
        message1 = '较好,'
    else:
        message1 = '非常好,'

    #饱和度分数
    hueScore = image_colorfulness(image)
    #类型
    high_probability_type, high_probability = orig_score(image)
    if hueScore < 50:
        message2 = '饱和度较低,'
    else:
        message2 = '饱和度较高,'
    message3 = high_probability_type

    return '该图片色彩表现' + message1 + message2 + '色彩搭配类型大概率属于' + message3

def single_evaluate(image):
    # image = default_loader(image_path)
    # image = image.resize((224, 224))
    hue_score = image_colorfulness(image)  
    image = TransformPicture(image.copy())
    image = torch.unsqueeze(image, 0)
    model = Color_Net()
    model.eval()
    model_path = './models/Color_Net_v2/color_net/Spaq_color_Mos_Model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    out = model(image)
    score, _ = get_score(out)
    score = re_score(score)
    color_score = enhance_score(score.item(), hue_score)
    return color_score

def multi_evaluate(image_file_path, copy_image_file_path, output_csv):
    with open(output_csv, 'a') as f:
        f.write("ID")
        f.write(',score')
        f.write(',color_wheel')
        f.write(',probability' + '\n')

    for filename in os.listdir(image_file_path):
        image = default_loader(image_file_path + "/" + filename)
        # image = image.resize((224, 224))
        image = TransformPicture(image)
        hue_score = image_colorfulness(image_file_path + "/" + filename)
        high_probability_type, high_probability = orig_score(image_file_path + "/" + filename)
        image = torch.unsqueeze(image, 0)
        out = model(image)
        score, _ = get_score(out)
        score = re_score(score)
        color_score =  enhance_score(score.item(), hue_score, high_probability)
        # new_name_str = str(round(score.item(), 3)) + "_" + str(high_probability_type) + "_" + str(high_probability)

        # 下面这个注释打开，复制所有待测图片，并以得分作为命名
        # shutil.copy(os.path.join(image_file_path, filename), os.path.join(copy_image_file_path, f"{new_name_str}"))

        # output to file
        with open(output_csv, 'a') as f:
            # 下面这句话，主要是截断文件名中的中文，防止乱码
            filename = filename.encode('UTF-8', 'ignore').decode('UTF-8')
            f.write(filename)
            f.write(',' + str(round(color_score, 2)))
            f.write(',' + str(high_probability_type))
            f.write(',' + "{:.2f}%".format(high_probability))
            f.write('\r\n')

    f.close()
    print("Complete!")

def getColorHarm(image_filename):
	color_image = cv2.imread(image_filename, cv2.IMREAD_COLOR)

	height = color_image.shape[0]
	width  = color_image.shape[1]

	HSV_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
	best_harmomic_scheme = color_harmonization.B(HSV_image)
	# print("Harmonic Scheme Type  : ", best_harmomic_scheme.m)
	# print("Harmonic Scheme Alpha : ", best_harmomic_scheme.alpha)
	return best_harmomic_scheme.alpha


class Color_Net(nn.Module):
    def __init__(self, pretrained_base_model=False):
        super(Color_Net, self).__init__()
        base_model = mobile_net_v2(pretrained=pretrained_base_model)
        base_model = nn.Sequential(*list(base_model.children())[:-1])

        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.75),
            nn.Linear(1280, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

if __name__ == '__main__':
    model = Color_Net()
    model.eval()
    model_path = './Spaq_color_Mos_Model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # model.to('cuda')

    # # 评测单张图片
    # image_path = "./test_img/4.jpeg"
    # score = single_evaluate(image_path)
    # hue_score = image_colorfulness(image_path)
    # high_probability_type, high_probability = orig_score(image_path)        
    # color_score =  enhance_score(score.item(), hue_score, high_probability)
    # print("色环搭配:", high_probability_type, "概率: {:.2f}%".format(high_probability))
    # print("色彩分数: {:.2f}".format(color_score))


    """
    评测多张图片
    输出方式：
    （1）结果写入scv文件中
    （2）复制所有待测图片，并以得分重命名
    输出格式：美学评分+美学评估可信概率
    """
    # 结果输出至哪个csv文件
    output_csv = './result.csv'
    if (os.path.isfile(output_csv) == False):
        open(output_csv, 'w')
    else:
        # 清空原来csv中的内容
        with open(output_csv, 'r+') as file:
            file.truncate(0)

    # 待评测图片所在文件夹
    image_file_path = './test_img'

    # 输出的重命名图片文件夹目录
    copy_image_file_path = './test_img_new'
    if (os.path.exists(copy_image_file_path) == False):
        os.makedirs(copy_image_file_path)

    # 评测一批图片
    multi_evaluate(image_file_path, copy_image_file_path, output_csv)
