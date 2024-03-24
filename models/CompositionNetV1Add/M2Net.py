import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "."))

import torch.nn as nn
from .mv2 import mobile_net_v2
import torch
import numpy as np
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import scipy.optimize as opt
import matplotlib
from matplotlib.font_manager import _rebuild
_rebuild() #reload一下
import matplotlib.pyplot as plt
import os

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
            nn.Linear(10,9),
            nn.Sigmoid()
            #nn.Softmax(dim=1)
        )


    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


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
    plt.clf()
    # w = torch.from_numpy(np.linspace(1, 10, 10))
    # w = w.type(torch.FloatTensor)
    # w = w.to(opt.device)
    res = ['三分', '水平', '垂直', '对角', '曲线', '三角', '中心', '对称', '模式']
    score_np = y_pred.data.cpu().numpy()
    # 这两行代码解决 plt 中文显示的问题
    # plt.rcParams['font.sas-serif'] = ['simhei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # print(matplotlib.matplotlib_fname())

    plt.rcParams['font.sans-serif'] = ['simhei']
    plt.rcParams['axes.unicode_minus'] = False

    classification = ('三分', '水平', '垂直', '对角', '曲线', '三角', '中心', '对称', '模式')
    prob = score_np[0] * 10
    plt.bar(classification, prob)
    plt.title('构图分类')
    plt.yticks([t for t in range(0, 12, 2)])
    res = ['三分', '水平', '垂直', '对角', '曲线', '三角', '中心', '对称', '模式']
    score_np = y_pred.data.cpu().numpy()
    if score_np[0][score_np.argmax()]<0.3:
        return 0,"No",plt.gcf()
    # print('1',score_np[0][score_np.argmax()])
    # print('2',res[score_np.argmax()])
    # print('3',plt.gcf())
    # plt.show()
    return score_np[0][score_np.argmax()],res[score_np.argmax()],plt.gcf()
def single_evaluate(image,model):
    # image = image.resize((224, 224))
    image = TransformPicture(image)
    image = torch.unsqueeze(image, 0).to('cuda')
    out = model(image)
    score, c,img = get_score(out)
    return score,c,img

def singleTest(image):
    # image_path = "./6.113_0.629.jpg"
    model = M_MNet()
    model.eval()

    model_path = current_dir + '/epoch_3631_0.22988655129481947.pth'

    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.to('cuda')
    score,c,img= single_evaluate(image, model)
    return score,c,img


def getMessage(image,score):
    message1 = ''
    message2 = ''
    if score < 3:
        message1 = '构图较差,'
    elif score >= 3 and score < 5:
        message1 = '构图一般,'
    elif score >= 5 and score < 7:
        message1 = '构图较佳,'
    else:
        message1 = '构图合理，和谐美观'
    scoreC,c,img = singleTest(image)
    if scoreC<0.4:
        message2 = '无明显构图'
    elif scoreC<0.6:
        message2 = '小概率为：'+str(c)+'构图'
    elif scoreC<0.7:
        message2 = '存在一定的：'+str(c)+'构图'
    elif scoreC>=0.7:
        message2 = '大概率为：'+str(c)+'构图'
    return message1+message2,img
if __name__ =="__main__":
    singleTest('f946839a13e8fb21559a1a01b42b067cplt.jpeg')