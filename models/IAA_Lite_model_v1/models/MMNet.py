# coding:utf-8
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "."))

import torch.nn as nn
from mv2 import mobile_net_v2
import torch
import os, shutil
# import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import numpy as np

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
    w = w.to('cuda')

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

def pre_probability(score, acc_0_3=1, acc_3_4=0.98, acc_4_5=0.80, acc_5_6=0.66, acc_6_7=0.94, acc_7_10=1.0):
    if score < 3:
        probability1 = acc_0_3 / 2
    elif 3 <= score < 4:
        probability1 = acc_3_4 / 2
    elif 4 <= score < 5:
        probability1 = acc_4_5 / 2
    elif 5 <= score < 6:
        probability1 = acc_5_6 / 2
    elif 6 <= score < 7:
        probability1 = acc_6_7 / 2
    elif 7 <= score <= 10:
        probability1 = acc_7_10 / 2
    return probability1

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

def single_evaluate(image):
    print(current_dir)
    model = M_MNet()
    model.eval()
    model_path = current_dir+'/best.pth'
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.to('cuda')
    # image = image.resize((224, 224))
    image = TransformPicture(image)
    image = torch.unsqueeze(image, 0).to('cuda')
    out = model(image)
    score, _ = get_score(out)
    probability2 = torch.max(out, dim=1)[0]
    if probability2 >= 0.48:
        probability2 = 0.48

    score_p = pre_probability(score) + probability2
    score = re_score(score)


    return score, score_p

def multi_evaluate(image_file_path, copy_image_file_path, output_csv):
    with open(output_csv, 'a') as f:
        f.write("ID")
        f.write(',score')
        f.write(',probability' + '\n')

    for filename in os.listdir(image_file_path):
        image = default_loader(image_file_path + "/" + filename)
        # image = image.resize((224, 224))
        image = TransformPicture(image)
        image = torch.unsqueeze(image, 0).to('cuda')
        out = model(image)
        score, _ = get_score(out)

        probability2 = torch.max(out, dim=1)[0]
        if probability2 >= 0.48:
            probability2 = 0.48

        score_p = pre_probability(score) + probability2
        score = re_score(score)

        if (isinstance(score_p, float)):
            score_p = torch.tensor([score_p], dtype=torch.float)
        new_name_str = str(round(score.item(), 3)) + "_" + str(round(score_p.item(), 3))

        # 下面这个注释打开，复制所有待测图片，并以得分作为命名
        shutil.copy(os.path.join(image_file_path, filename), os.path.join(copy_image_file_path, f"{new_name_str}.jpg"))

        # output to file
        with open(output_csv, 'a') as f:
            # 下面这句话，主要是截断文件名中的中文，防止乱码
            filename = filename.encode('UTF-8', 'ignore').decode('UTF-8')
            f.write(filename)
            f.write(',' + str(score.item()))
            # f.write(',' + str(score_p.item()) + '\n')
            f.write(',' + str(score_p.item()) + '\n')

    f.close()
    print("Complete!")

class M_MNet(nn.Module):
    def __init__(self, pretrained_base_model=False):
        super(M_MNet, self).__init__()
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
    model = M_MNet()
    model.eval()
    model_path = './1_srcc_best_balance_data_distort_goodaddscore_back_AVA_balance_remove_unusual_1_resocre-3-7-2_vacc0.8283796740172579_srcc0.8122795303100221.pth'
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.to('cuda')

    # 评测单张图片
    # image_path = "./6.113_0.629.jpg"
    # score, score_p = single_evaluate(image_path)
    # print("score:{}, probability:{}".format(score.item(), score_p.item()))

    """
    评测多张图片
    输出方式：
    （1）结果写入scv文件中
    （2）复制所有待测图片，并以得分重命名
    输出格式：美学评分+美学评估可信概率
    """
    # # 结果输出至哪个csv文件
    output_csv = './result.csv'
    if (os.path.isfile(output_csv) == False):
        open(output_csv, 'w')
    else:
        # 清空原来csv中的内容
        with open(output_csv, 'r+') as file:
            file.truncate(0)

    # 待评测图片所在文件夹
    image_file_path = '/root/tmp/pycharm_project_815/M_M_Semi-Supervised/code/AVA/models/test_image_score'

    # 输出的重命名图片文件夹目录
    copy_image_file_path = './test_8_13_score/'
    if (os.path.exists(copy_image_file_path) == False):
        os.makedirs(copy_image_file_path)

    # 评测一批图片
    multi_evaluate(image_file_path, copy_image_file_path, output_csv)
