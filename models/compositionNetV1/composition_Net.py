from __future__ import division
import pandas as pd
# from keras.optimizers import Adam

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "."))

import sys, os,shutil
import csv
from .utilitiestrain import preprocess_imagesandsaliencyforiqa

from .modelfinal import TVdist, SGDNet
import h5py, yaml

from argparse import ArgumentParser

# import keras.backend as K

# def mean_pred(y_true, y_pred):
#     return K.mean(y_pred)

# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy', mean_pred])
from scipy import stats
def rename_pics(files_temp,copy_image_file_path,score0):
    count=0
    file_array=os.listdir(files_temp)
    file_array.sort()
    for info in file_array:
        picname=info.split('.j')[-2]
        old_picname=os.path.join(files_temp,info)#
        temp_picname=picname+"_N_{}".format(score0[count])+'.jpg'
        new_picname=os.path.join(copy_image_file_path,temp_picname)
        shutil.copy(old_picname,new_picname)
        count+=1
def single_evaluate(image_path,size):
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    model = SGDNet(basemodel='resnet', saliency='output', CA='true', fixed='False', img_cols=512,
                   img_rows=384, out2dim=1024)

    model.summary()
    arg = 'saliencyoutput-alpha0.25-ss-AVA-1024-EXP0-lr=0.0001-bs=15.23-0.4174-0.1842-0.4320-0.3593.h5'
    print("Load weights SGDNet")
    weight_file = current_dir + '/checkpoint/' + arg
    model.load_weights(weight_file)

    x, x1 = preprocess_imagesandsaliencyforiqa(
        [image_path],
        [''], 384, 512, crop_h=384, crop_w=512)
    res = model(x)
    score = res[0].numpy()[0][0]
    t = '%.2f' % score
    return t
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
        score = single_evaluate(img_path_list[i],256)
        ID.append(img_path_list[i])
        scores.append(score)
    data = pd.DataFrame({"ID": ID, "score": scores})
    data.to_csv(output_csv, index=False)
    # 下面这个注释打开，复制所有待测图片，并以得分作为命名
    rename_pics(image_file_path, copy_image_file_path,scores)
    print("Complete!")

if __name__ == '__main__':
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    model = SGDNet(basemodel='resnet', saliency='output', CA='true', fixed='False', img_cols=512,
                   img_rows=384, out2dim=1024)

    model.summary()
    arg = 'saliencyoutput-alpha0.25-ss-AVA-1024-EXP0-lr=0.0001-bs=15.23-0.4174-0.1842-0.4320-0.3593.h5'
    print("Load weights SGDNet")
    weight_file = './checkpoint/' + arg
    model.load_weights(weight_file)

    #测单张图片
    image_path = "./Test_img/63fb85d759220c53bb812579b06dd9d6.jpeg"
    score = single_evaluate(image_path, 256)
    print("score:{}".format(score))
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