from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from Model_ViT import Multi_Label_ViT_Net
import torch
import cv2
import numpy as np


def reshape_transform(tensor, height=14, width=14):
    '''
    reshape output from transformer

    :param tensor: transformer output
    :param height: height of image
    :param width: width of image

    :return: reshaped tensor
    '''
    # 去掉cls token
    result = tensor[:, 1:, :]
    result = result.reshape(tensor.size(0), height, width, tensor.size(2))

    # 将通道维度放到第一个位置
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_grad_cam(model, image_path, image_size=224):
    '''
    获取 Grad-CAM 图

    :param model: 模型
    :param image_path: 图像路径

    :return: Grad-CAM 图(cv2)
    '''
    model.eval()

    # 创建 GradCAM 对象
    cam = GradCAM(model=model.encoder, target_layers=[model.encoder.encoder.layers.encoder_layer_22.ln_2],
                  reshape_transform=reshape_transform)

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    origin_image_size = (rgb_img.shape[1], rgb_img.shape[0])
    rgb_img = cv2.resize(rgb_img, (image_size, image_size))

    # 预处理图像
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 计算 grad-cam
    target_category = None  # 可以指定一个类别，或者使用 None 表示最高概率的类别
    grayscale_cam = cam(input_tensor=input_tensor, targets=target_category)
    grayscale_cam = grayscale_cam[0, :]

    # 将 grad-cam 的输出叠加到原始图像上
    rgb_img = np.float32(rgb_img) / 255
    visualization = show_cam_on_image(rgb_img, grayscale_cam)

    # 提高分辨率
    cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    visualization = cv2.resize(visualization, origin_image_size)

    return visualization


# if __name__ == '__main__':
#     hidden_dim = 1024
#     fusion_num_heads = 16
#     image_path = 'test_images/C_2144_left.jpg'
#     pretrain_model_path = 'model_weights/best_Multi_label_ViT_Net_1.pth'
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     model = Multi_Label_ViT_Net(hidden_dim=hidden_dim, fusion_num_heads=fusion_num_heads).to(device)
#     model.load_state_dict(torch.load(pretrain_model_path))
#
#     grad_cam = get_grad_cam(model, image_path)
#     # 保存结果
#     cv2.imwrite('grad-cam.jpg', grad_cam)
