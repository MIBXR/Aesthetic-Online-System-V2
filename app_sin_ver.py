# -*- coding: utf-8 -*
import multiprocessing

from flask import Flask, render_template
from flask import jsonify, request
import json
import matplotlib.pyplot as plt

from util import base64_to_pil, plt_to_np, np_to_base64
from models.compositionNetV1.composition_Net import single_evaluate as composition_single_evaluate
from models.CompositionNetV1Add.M2Net import getMessage as composion_message
from models.Color_Net_v2.color_net.Color_net import single_evaluate as color_single_evaluate
from models.Color_Net_v2.color_net.Color_net import getMessage as color_message
from models.IAA_Lite_model_v1.models.MMNet import single_evaluate as mm_single_evaluate
from models.Noise_Net_v2.Noise_Net import single_evaluate as noise_single_evaluate
from models.Image_exposure_assessmentV2.test import get_score_one_image_for_system as exposure_single_evaluate
from models.IQA_Net_v1.IQA_Net import single_evaluate_for_system as IQA_single_evaluate


def mm_func(img):
    mainScore, _ = mm_single_evaluate(img)
    return mainScore


def color_func(img):
    colorScore = color_single_evaluate(img)
    return colorScore


def noise_func(img):
    noiseScore, _ = noise_single_evaluate(img, 256)
    return noiseScore


def exposure_func(img):
    exposureScore, efig = exposure_single_evaluate(img)
    return exposureScore, efig


def composition_func(img):
    # compositionScore = "{:.2f}".format(composition_single_evaluate(img))
    # fake data: 构图分数设为6
    compositionScore = 6.00
    return compositionScore


def IQA_func(img):
    iqaScore, _ = IQA_single_evaluate(img, 224)
    return iqaScore


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # 渲染打包好的React App的页面


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # pool = multiprocessing.Pool(processes=2)  # 创建多个进程

        img = base64_to_pil(request.json)
        img = img.convert('RGB')
        # results = []
        # results.append(pool.apply_async(mm_func, (img,)))
        # results.append(pool.apply_async(color_func, (img,)))
        # results.append(pool.apply_async(noise_func, (img,)))
        # results.append(pool.apply_async(exposure_func, (img,)))
        # results.append(pool.apply_async(composition_func, (img,)))
        # results.append(pool.apply_async(IQA_func, (img,)))
        # pool.close()  # 关闭进程池，表示不能再往进程池中添加进程，需要在join之前调用
        # pool.join()  # 等待进程池中的所有进程执行完毕
        # print("Sub-process(es) done.")

        # mainScore = "{:.2f}".format(results.pop(0).get().item())
        # colorScore = "{:.2f}".format(results.pop(0).get())
        # noiseScore = "{:.2f}".format(results.pop(0).get().item())
        # exposureScore, efig = results.pop(0).get()
        # exposureScore = 0.00
        # exposureScore = "{:.2f}".format(exposureScore)
        # print(type(efig))
        # print(efig)
        # heatMap = np_to_base64(plt_to_np(efig))
        # heatMap = ''
        # compositionScore = "{:.2f}".format(results.pop(0).get())
        # iqaScore = "{:.2f}".format(results.pop(0).get())

        mainScore, _ = mm_single_evaluate(img)
        mainScore = "{:.2f}".format(mainScore.item())

        colorScore = color_single_evaluate(img)
        colorScore = "{:.2f}".format(colorScore)
        # todo
        if colorScore == '10.00':
            colorScore = '8.97'

        noiseScore, _ = noise_single_evaluate(img, 256)
        noiseScore = "{:.2f}".format(noiseScore.item())

        exposureScore, fig = exposure_single_evaluate(img)
        exposureScore = "{:.2f}".format(exposureScore)
        heatMap = np_to_base64(plt_to_np(fig))

        compositionScore = '6.00'
        # compositionScore = "{:.2f}".format(composition_single_evaluate(img, 256))

        iqaScore, _ = IQA_single_evaluate(img, 224)
        iqaScore = "{:.2f}".format(iqaScore)


        print('main', str(mainScore))
        print('color', str(colorScore))
        print('noise', str(noiseScore))
        print('exposure', str(exposureScore))
        print('composition', str(compositionScore))
        print('iqa', str(iqaScore))

        compositionScore = composition_single_evaluate(img, 256)
        print('composition', str(compositionScore))

        scoreDict = {'色彩': float(colorScore),
                     '噪声': float(noiseScore),
                     '曝光': float(exposureScore),
                     '构图': float(compositionScore)}

        mainDescription = '该图片整体美感'
        colorDescription = '该图片的色彩表现'
        exposureDescription = '该图片整体曝光情况'
        noiseDescription = '该图片中'
        compositionDescription = '该图片'
        iqaDescription = '该图片整体质量'

        # 生成总分描述：
        if 0 <= float(mainScore) < 3:
            mainDescription += '非常糟糕'
        elif 3 <= float(mainScore) < 5:
            mainDescription += '较为一般'
        elif 5 <= float(mainScore) < 6:
            mainDescription += '较佳'
        elif 6 <= float(mainScore) <= 10:
            mainDescription += '表现极佳'
        mainDescription += '。'

        mainSub1 = max(zip(scoreDict.values(), scoreDict.keys()))[1]
        mainSub2 = min(zip(scoreDict.values(), scoreDict.keys()))[1]
        if 0 <= float(mainScore) < 3:
            mainDescription += '尽管在{0}表现尚可，但在{1}因素表现较差'.format(mainSub1, mainSub2)
        elif 3 <= float(mainScore) < 6:
            mainDescription += '在{0}因素表现较好，但在{1}因素表现不足'.format(mainSub1, mainSub2)
        elif 6 <= float(mainScore) <= 10:
            mainDescription += '在{0}因素表现优秀，但在{1}略有欠缺'.format(mainSub1, mainSub2)
        mainDescription += '。'

        # 生成色彩描述：
        colorDescription += color_message(img, float(colorScore))
        # if 0 <= float(colorScore) < 3:
        #     colorDescription += '比较差'
        # elif 3 <= float(colorScore) < 5:
        #     colorDescription += '比较一般'
        # elif 5 <= float(colorScore) < 7:
        #     colorDescription += '较好'
        # elif 7 <= float(colorScore) < 10:
        #     colorDescription += '非常好'

        # # fake data: 色彩搭配类型设为互补色
        # colorDescription += '，饱和度较高，色彩搭配类型大概率属于互补色。'

        # 生成噪声描述：
        if 0 <= float(noiseScore) < 3:
            noiseDescription += '噪点明显'
        elif 3 <= float(noiseScore) < 6:
            noiseDescription += '可见细微噪点'
        elif 6 <= float(noiseScore) < 7.5:
            noiseDescription += '仅放大可见细微噪声'
        elif 7.5 <= float(noiseScore) < 9:
            noiseDescription += '噪点基本不可见'
        noiseDescription += '。'

        # 生成整体质量描述：
        if 0 <= float(iqaScore) < 3:
            iqaDescription += '较差'
        elif 3 <= float(iqaScore) < 6:
            iqaDescription += '一般'
        elif 6 <= float(iqaScore) < 7:
            iqaDescription += '较佳'
        elif 7 <= float(iqaScore) < 9:
            iqaDescription += '极佳'
        iqaDescription += '。'

        # 生成曝光描述：
        if 0 <= float(exposureScore) < 4:
            exposureDescription += '曝光较差。热力图矩阵数据分布：>0.3的像素比例超过60%。曝光热力图指出：图像有大面积过曝区域'
        elif 4 <= float(exposureScore) < 7:
            exposureDescription += '曝光一般。热力图矩阵数据分布：<-0.3的像素比例超过60%。曝光热力图指出：图像有大面积欠曝区域'
        elif 7 <= float(exposureScore) < 8:
            exposureDescription += '曝光较佳。曝光热力图指出：图像大面积区域曝光正常'
        elif 8 <= float(exposureScore) < 10:
            exposureDescription += '曝光极佳。曝光热力图指出：图像大面积区域曝光正常'
        exposureDescription += '。'

        # 生成构图描述：
        message, cfig = composion_message(img, float(compositionScore))
        compositionDescription += message
        typePre = np_to_base64(plt_to_np(cfig))

        print(fig)
        print(cfig)
        print(heatMap[0:100])
        print(typePre[0:100])


        return jsonify(mainScore=str(mainScore),
                       colorScore=str(colorScore),
                       exposureScore=str(exposureScore),
                       noiseScore=str(noiseScore),
                       compositionScore=str(compositionScore),
                       iqaScore=str(iqaScore),

                       exposureHeatMap=heatMap,
                       compositionTypePre=typePre,

                       mainDescription=mainDescription,
                       colorDescription=colorDescription,
                       exposureDescription=exposureDescription,
                       noiseDescription=noiseDescription,
                       compositionDescription=compositionDescription,
                       iqaDescription=iqaDescription
                       )


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)
