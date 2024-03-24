import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_colorfulness(image): 
    # image = cv2.imread(img_path)
    #将图片分为B,G,R三部分（注意，这里得到的R、G、B为向量而不是标量） 
    image = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    (B, G, R) = cv2.split(image) 

    #rg = R - G
    rg = np.absolute(R - G) 

    #yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B) 

    #计算rg和yb的平均值和标准差
    (rbMean, rbStd) = (np.mean(rg), np.std(rg)) 
    (ybMean, ybStd) = (np.mean(yb), np.std(yb)) 

    #计算rgyb的标准差和平均值
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2)) 

    # 返回颜色丰富度C 
    return stdRoot + (0.3 * meanRoot)

if __name__ == '__main__':
    #测试单张图片
    img_path = "/Users/lijialong/Desktop/IAA/test_img/15264301581631170.jpg"
    # img = cv2.imread(img_path)
    print(image_colorfulness(img_path))

    #批量测试图片
    # path = '/home/ljl/color_code/ReLIC-master/code/AVA/color_test'
    # files = os.listdir(path)
    # for filename in files:
    #     if filename == '.DS_Store':
    #         continue
    #     print(filename)
    #     img_path = path + '/' + filename
    #     print('hue score : ', image_colorfulness(cv2.imread(img_path)))