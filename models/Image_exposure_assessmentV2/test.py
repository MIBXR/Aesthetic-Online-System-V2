import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "."))
import seaborn as sns
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import glob
from PIL import Image
import scipy.ndimage
import torch
import time
import numpy as np
from torchvision import transforms
# from thop import profile
from model.UNet_2Plus import UNet_2Plus
from model.unet3plus import UNet_3Plus
from model.UNet import UNet
from torchvision.datasets.folder import default_loader
device = torch.device("cuda:0")

def plot_heatmap_then_save(Img_pred, h,w,save_dir,score,exp_level,img_name='test1',is_save = True, is_plot = True):
    #根据网络预测的结果绘制曝光评价热力图
    sns.heatmap(-Img_pred, cmap='jet' ,vmax=0.5, vmin=-0.7, center=0, robust=True,xticklabels=False,yticklabels=False)
    fig = plt.gcf()
    if is_plot == True:
        #是否要画出plot
        plt.show()

    if is_save == True:
        #是否保存热力图到本地文件夹
        save_path = save_dir + img_name + exp_level+'_' + f'{round(score,3)}'+'.png'
        fig.savefig(save_path)


def plot_heatmap_for_system(Img_pred):
    #根据网络预测的结果绘制曝光评价热力图
    sns.heatmap(-Img_pred, cmap='jet' ,vmax=0.5, vmin=-0.7, center=0, robust=True,xticklabels=False,yticklabels=False)
    # plt.show()
    fig = plt.gcf()
    return fig

def TransformPicture(x):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])
    return transform(x)

def load_pic(img_path):
    image = default_loader(img_path)
    w,h = image.size
    image = TransformPicture(image)
    image = torch.unsqueeze(image, 0).to(device)
    return image,w,h

def get_score_one_image_for_system(img):
    model_weight_path = 'models/Image_exposure_assessmentV2/weights/UNet++_43all_leakyrelu1_l1loss_continue_2021_12_14_02_55_epoch_22_tloss_0.080023773001644_vloss0.3391493246019856.pth'
    model = UNet_2Plus()
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))  # 读取权重
    model.to(device)
    test_img = TransformPicture(img)
    test_img = torch.unsqueeze(test_img, 0).to(device)
    pred = model(test_img).to('cpu')
    pred = pred.detach().squeeze()
    score = 20 * (0.5 - np.array(pred.abs()).mean())
    fig = plot_heatmap_for_system(pred)
    return score, fig

if __name__ == '__main__':
    model_weight_path = './weights/UNet++_43all_leakyrelu1_l1loss_continue_2021_12_14_02_55_epoch_22_tloss_0.080023773001644_vloss0.3391493246019856.pth'
    save_dir = './test_result/'
    #测试单张图像
    cpuDevice = torch.device('cpu')
    test_img_path = 'C:/Users/catalyst/Desktop/Aesthetic-Online-System/models/Image_exposure_assessmentV2/test_imgs/11_P30Pro-3.jpg'
    model = UNet_2Plus()
    model.load_state_dict(torch.load(model_weight_path, map_location='cuda:0'))  # 读取权重
    model.to(cpuDevice)
    test_img,w,h = load_pic(test_img_path)
    pred = model(test_img).to('cpu')
    pred = pred.detach().squeeze()
    score = 20 * (0.5 - np.array(pred.abs()).mean())
    plot_heatmap_then_save(pred, h, w, save_dir, score, exp_level='_UNet2+_3Layer')
    print('score',score)

    #测试多张图像
    # test_img_dir = './test_imgs/'
    # model = UNet_2Plus()
    # model.load_state_dict(torch.load(model_weight_path, map_location='cuda:0'))
    # model.to(device)
    # #测试文件夹内的多张jpg图像:
    # file_list = glob.glob(test_img_dir + '*.jpg')
    # for i in range(len(file_list)):
    #     img_name = file_list[i].split('\\')[-1].split('.')[0]
    #     test_img, w, h = load_pic(file_list[i])
    #     pred = model(test_img).to('cpu')
    #     pred = pred.detach().squeeze()
    #     pred_abs = pred.abs()
    #     score = 20 * (0.5 - np.array(pred.abs()).mean())
    #     plot_heatmap_then_save(pred, h, w, save_dir, score,img_name=img_name, exp_level='_UNet2+_3Layer')
        #这个score是曝光质量评分


    #计算模型参数量
    # net=UNet_3Plus()
    # net.load_state_dict(torch.load(model_weight_path, map_location='cuda:0'))
    #
    # #采用这一个方法：
    # from ptflops import get_model_complexity_info
    # flops, params = get_model_complexity_info(net.to(device), (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    # print('Flops:  ' + flops)
    # print('Params: ' + params)

    #方法二
    # from thop import profile
    # x = torch.randn(1, 3,128, 128)
    # flops, params = profile(net, inputs=(x,))
    # print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    # print("params=", str(params / 1e6) + '{}'.format("M"))

    #方法三：
    #from torchstat import stat
    #stat(net, (3, 256, 256))

    #方法四：
    #from torchsummary import summary
    #summary(net.to(device), (3, 256, 256))






