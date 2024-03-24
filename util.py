"""Utilities
"""
import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO


def base64_to_pil(img_base64):
    """
    Convert base64 image data to PIL image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):
    """
    Convert numpy image (RGB) to base64 string
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

def plt_to_np(fig):
    '''
    将matplotlib图像转换为numpy.array
    '''
    # # 将plt转化为numpy数据
    # canvas = FigureCanvasAgg(plt.gcf())
    # # 绘制图像
    # canvas.draw()
    # # 获取图像尺寸
    # w, h = canvas.get_width_height()
    # # 解码string 得到argb图像
    # buf = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)

    # 绘制图像
    fig.canvas.draw()
    # 获取图像尺寸
    w, h = fig.canvas.get_width_height()
    # 获取 argb 图像
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)

    # 重构成w h 4(argb)图像
    buf.shape = (w, h, 4)
    # 转换为 RGBA
    buf = np.roll(buf, 3, axis=2)
    # 得到 Image RGBA图像对象
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    # 转换为numpy array rgba四通道数组
    image = np.asarray(image)
    # 转换为rgb图像
    rgb_image = image[:, :, :3]
    return rgb_image