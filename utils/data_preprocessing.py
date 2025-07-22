import cv2
import torchvision.transforms as transforms
import numpy as np
from skimage.filters import frangi
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Normalize


# GCN
def global_contrast_normalization(image, mean=None, std=None):
    '''
    全局对比度归一化

    :param image: 图像(tensor)
    :param mean: 均值
    :param std: 方差

    :return: 归一化后的图像
    '''
    image = image.to(torch.float32)
    if mean is None:
        mean = image.mean(dim=(0, 1), keepdim=True)
    if std is None:
        std = image.std(dim=(0, 1), keepdim=True)

    return (image - mean) / (std + 1e-5)

# RGB to Lab
def rgb_to_lab(image):
    '''
    RGB to Lab

    :param image: RGB 图像(tensor)

    :return: Lab 图像
    '''
    # image = image.permute(1, 2, 0).cpu().numpy()        # 转换为 HxWxC 格式
    lab_image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2LAB)

    # return lab_image.permute(2, 0, 1)  # 转换回 CxHxW 格式
    return torch.tensor(lab_image)

# Scaling
def scaling(image, scaled_image_size):
    '''
    缩放

    :param image: 图像(tensor)
    :param scaled_image_size: 缩放后图像尺寸

    :return: 缩放后图像
    '''
    image = image.permute(2, 0, 1)      # 转换为 CxHxW 格式
    image = transforms.Resize(scaled_image_size)(image)

    return image.permute(1, 2, 0)      # 转换回 HxWxC 格式

# CLAHE
def clahe(image):
    '''
    有限对比度自适应直方图均衡化

    :param image: 图像(tensor, Lab 格式)

    :return: 均衡化后的图像
    '''
    l, a, b = cv2.split(image.numpy())      # 图像应为 Lab
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    return torch.tensor(image)

# Bilateral filter
def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75):
    '''
    双边滤波

    :param image: 图像
    :param d: 邻域直径
    :param sigma_color: 颜色空间参数
    :param sigma_space: 坐标空间参数

    :return: 滤波后的图像
    '''
    image = cv2.bilateralFilter(image.numpy(), d, sigma_color, sigma_space)

    return torch.tensor(image)

# Merge image
def merge_image(image_l, image_r):
    '''
    横向拼接两张图像

    :param image_l: 左图像(应为 PIL.Image)
    :param image_r: 右图像(应为 PIL.Image)

    :return: 拼接后的图像
    '''
    # 获取两张图片的大小
    width_l, height_l = image_l.size
    width_r, height_r = image_r.size

    # 创建一个新的空白图片，宽度是两张图片宽度之和
    new_width = width_l + width_r
    new_height = height_l
    new_image = Image.new("RGB", (new_width, new_height))  # 创建新图像

    # 将图片粘贴到新图像中
    new_image.paste(image_l, (0, 0))
    new_image.paste(image_r, (width_l, 0))

    return new_image

# Split image
def split_image(image):
    '''
    横向拆分图像

    :param image: 图像(应为 PIL.Image)

    :return: 左图像和右图像
    '''
    # 获取图片的大小
    width, height = image.size

    # 拆分图片
    left_image = image.crop((0, 0, width // 2, height))
    right_image = image.crop((width // 2, 0, width, height))

    return left_image, right_image

# Segment normalization
def segment_normalization(imgs_list):
    '''
    用于分割任务的图像归一化

    :param imgs_list: 图像列表

    :return: 归一化后的图像列表
    '''
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list

# if __name__ == '__main__':
    # test classification data preprocessing

    # test_image_path = './test_images/1_left.jpg'
    # origin_image = Image.open(test_image_path)
    #
    # plt.imshow(origin_image)
    # plt.title('Origin')
    # plt.show()
    #
    # test_image = np.array(origin_image)
    # test_image = torch.tensor(test_image)
    #
    # plt.imshow(test_image)
    # plt.title('Test')
    # plt.show()
    #
    #
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # # test GCN
    # # 处理后图像
    # solved_image = global_contrast_normalization(test_image)
    #
    # # test RGB to Lab
    # solved_image = rgb_to_lab(solved_image)

    # # test Scaling
    # scaled_image_size = (256, 256)
    # solved_image = scaling(test_image, scaled_image_size)

    # # test CLAHE
    # test_image = rgb_to_lab(test_image)
    # solved_image = clahe(test_image)

    # # test Bilateral filter
    # solved_image = bilateral_filter(test_image)

    # # test Frangi filter
    # solved_image = frangi_filter(test_image)

    # # test Dilation
    # solved_image = dilation(test_image)

    # # 原始图像
    # axes[0].imshow(origin_image)
    # axes[0].set_title('Origin')
    #
    # # 处理后图像
    # axes[1].imshow(solved_image, cmap='gray')
    # axes[1].set_title('Solved')
    #
    # plt.show()

    # # test merge and split image
    # test_image_path_l = 'test_images/1_left.jpg'
    # test_image_path_r = 'test_images/1_right.jpg'
    # origin_image_l = Image.open(test_image_path_l)
    # origin_image_r = Image.open(test_image_path_r)

    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    #
    # # 原始左眼图像
    # axes[0].imshow(origin_image_l)
    # axes[0].set_title('Origin left')
    #
    # # 原始右眼图像
    # axes[1].imshow(origin_image_r)
    # axes[1].set_title('Origin right')
    #
    # plt.show()

    # test_image_l = origin_image_l
    # test_image_r = origin_image_r

    # test merge image
    # merged_image = merge_image(test_image_l, test_image_r)
    # merged_image = np.array(merged_image)
    #
    # plt.imshow(merged_image)
    # plt.title('Merged')
    # plt.show()

    # # test split image
    # merged_image = merge_image(test_image_l, test_image_r)
    # image_l, image_r = split_image(merged_image)
    #
    # fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    #
    # # 拆分后左眼图像
    # axes[0].imshow(np.array(image_l))
    # axes[0].set_title('split left')
    #
    # # 拆分后右眼图像
    # axes[1].imshow(np.array(image_r))
    # axes[1].set_title('split right')
    #
    # plt.show()




