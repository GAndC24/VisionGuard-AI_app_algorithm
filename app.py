from flask import Flask, request, jsonify, url_for
import numpy as np
from Model_ViT import Multi_Label_ViT_Net
import torch
from PIL import Image
from utils.data_preprocessing import *
from werkzeug.utils import secure_filename
import os
import logging
from io import BytesIO
import requests
import uuid
from utils.grad_cam import get_grad_cam
from torchvision.transforms import Grayscale, ToTensor, Resize
from fr_unet import FR_UNet
from bunch import Bunch
import gc



app = Flask(__name__)
model_classify = None
model_segment = None
app.logger.setLevel(logging.DEBUG)  # 设置日志级别为 DEBUG
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 超参数
hidden_dim = 1024
fusion_num_heads = 16
img_size = 224
seg_img_size = 592
pretrained_model_path = 'model_weights/best_Multi_label_ViT_Net_1.pth'
segment_model_path = "model_weights/FR-UNet_DRIVE_checkpoint-epoch40.pth"
save_dir = 'static/saved'
temp_dir = 'static/temp'

# 配置参数
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # 允许的文件类型
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB

# 检查文件类型是否合法
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 加载分类和分割模型
def load_model():
    '''
    加载分类和分割模型
    '''
    # 设置环境变量，避免OOM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    global model_classify
    model_classify = Multi_Label_ViT_Net(hidden_dim=hidden_dim, fusion_num_heads=fusion_num_heads).to(device)
    model_classify.load_state_dict(torch.load(pretrained_model_path))
    model_classify.eval()
    app.logger.debug("Classify Model loaded successfully")

    global model_segment
    torch.serialization.add_safe_globals([Bunch])
    model_segment = FR_UNet().to(device)
    checkpoint = torch.load(segment_model_path, weights_only=True)
    state_dict = checkpoint['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}  # 去除 'module' 前缀
    model_segment.load_state_dict(state_dict)
    model_segment.eval()
    app.logger.debug("Segment Model loaded successfully")

# 保存图像并生成 URL
def save_and_generate_url(img):
    '''
    保存图像并生成URL
    Args:
        :param img: 图像(PIL.Image)
    Returns:
        :return: 图像URL
    '''
    img_filename = f"{uuid.uuid4().hex}.jpg"
    img_path = os.path.join(save_dir, img_filename)
    img.save(img_path, format='JPEG')
    img_url = url_for('static', filename=f'saved/{img_filename}', _external=True)
    return img_url

# 生成置信度图
def generate_confidence_map(confidence_map):
    '''
    生成置信度图并保存至 temp 目录

    :param confidence_map: 置信度图(numpy)

    :return: file name
    '''
    temp_img_filename = f"confidence_map_{uuid.uuid4().hex}.jpg"
    plt.imshow(confidence_map, cmap='jet')
    plt.axis('off')
    plt.savefig(f'{temp_dir}/{temp_img_filename}', bbox_inches='tight', pad_inches=0)
    plt.close()

    return temp_img_filename

@app.route('/predict', methods=['POST'])
def predict():
    # 检查请求中是否包含JSON数据
    if not request.is_json:
        return jsonify({'error': 'Request must be in JSON format'}), 400

    data = request.get_json()
    img_l_url_list = data.get('image_left_url_list')
    img_r_url_list = data.get('image_right_url_list')

    # 检查URL是否存在
    # check left
    for img_l_url in img_l_url_list:
        if not img_l_url:
            app.logger.debug(f"check img_l_url: {img_l_url}")
            return jsonify({'error': 'Missing image URLs in request'}), 400

    # check right
    for img_r_url in img_r_url_list:
        if not img_r_url:
            app.logger.debug(f"check img_r_url: {img_r_url}")
            return jsonify({'error': 'Missing image URLs in request'}), 400

    try:
        global model_classify
        global model_segment
        # 从 URL 获取图像
        # get left images
        img_l_list = []
        for img_l_url in img_l_url_list:
            response_l = requests.get(img_l_url, timeout=10)
            response_l.raise_for_status()       # 触发HTTP错误
            img_l = Image.open(BytesIO(response_l.content)).convert('RGB')
            img_l_list.append(img_l)

        # get right
        img_r_list = []
        for img_r_url in img_r_url_list:
            response_r = requests.get(img_r_url, timeout=10)
            response_r.raise_for_status()       # 触发HTTP错误
            img_r = Image.open(BytesIO(response_r.content)).convert('RGB')
            img_r_list.append(img_r)

        # 预处理 - classify
        classify_img_l_list = img_l_list
        classify_img_r_list = img_r_list

        # 预处理 - left
        classify_img_l_list = [np.array(img) for img in classify_img_l_list]
        classify_img_l_list = [torch.tensor(img) for img in classify_img_l_list]
        classify_img_l_list = [scaling(img, (img_size, img_size)) for img in classify_img_l_list]  # scaling
        classify_img_l_list = [rgb_to_lab(img) for img in classify_img_l_list]  # RGB to Lab
        classify_img_l_list = [clahe(img) for img in classify_img_l_list]  # CLAHE
        classify_img_l_list = [bilateral_filter(img) for img in classify_img_l_list]  # Bilateral filter

        # 预处理 - right
        classify_img_r_list = [np.array(img) for img in classify_img_r_list]
        classify_img_r_list = [torch.tensor(img) for img in classify_img_r_list]
        classify_img_r_list = [scaling(img, (img_size, img_size)) for img in classify_img_r_list]  # scaling
        classify_img_r_list = [rgb_to_lab(img) for img in classify_img_r_list]  # RGB to Lab
        classify_img_r_list = [clahe(img) for img in classify_img_r_list]  # CLAHE
        classify_img_r_list = [bilateral_filter(img) for img in classify_img_r_list]  # Bilateral filter

        # 合并
        merged_img_list = []
        for img_l, img_r in zip(classify_img_l_list, classify_img_r_list):
            img_l = img_l.numpy()
            img_r = img_r.numpy()
            img_l = Image.fromarray(img_l)
            img_r = Image.fromarray(img_r)
            merged_img = merge_image(img_l, img_r)
            merged_img_list.append(merged_img)

        # 保存并生成 URL
        merged_img_url_list = []
        for merged_img in merged_img_list:
            merged_img_url = save_and_generate_url(merged_img)
            merged_img_url_list.append(merged_img_url)

        # 获取 Grad - CAM 图
        grad_cam_img_l_list = []
        grad_cam_img_r_list = []
        for img_l, img_r in zip(classify_img_l_list, classify_img_r_list):
            img_l_save_path = f'{save_dir}/{uuid.uuid4().hex}.jpg'
            img_r_save_path = f'{save_dir}/{uuid.uuid4().hex}.jpg'
            img_l = img_l.numpy()
            img_r = img_r.numpy()
            img_l = Image.fromarray(img_l)
            img_r = Image.fromarray(img_r)
            img_l.save(img_l_save_path)
            img_r.save(img_r_save_path)

            grad_cam_img_l = get_grad_cam(model_classify, img_l_save_path)
            grad_cam_img_r = get_grad_cam(model_classify, img_r_save_path)
            grad_cam_img_l_list.append(grad_cam_img_l)
            grad_cam_img_r_list.append(grad_cam_img_r)

        # 合并 grad cam image 并生成 URL
        merged_grad_cam_img_url_list = []
        for grad_cam_img_l, grad_cam_img_r in zip(grad_cam_img_l_list, grad_cam_img_r_list):
            merged_grad_cam_img = merge_image(Image.fromarray(grad_cam_img_l), Image.fromarray(grad_cam_img_r))
            merged_grad_cam_img_url = save_and_generate_url(merged_grad_cam_img)
            merged_grad_cam_img_url_list.append(merged_grad_cam_img_url)

        # 预测 - classify

        # 归一化处理
        # 归一化 left
        classify_img_l_list = [global_contrast_normalization(img) for img in classify_img_l_list]  # GCN
        # 归一化 right
        classify_img_r_list = [global_contrast_normalization(img) for img in classify_img_r_list]  # GCN

        # 批量堆叠
        # left
        classify_img_l_list = [img.permute(2, 0, 1) for img in classify_img_l_list]  # 调整为 C, H, W
        classify_img_l_batch = torch.stack(classify_img_l_list, dim=0).to(device)
        # right
        classify_img_r_list = [img.permute(2, 0, 1) for img in classify_img_r_list]  # 调整为 C, H, W
        classify_img_r_batch = torch.stack(classify_img_r_list, dim=0).to(device)

        label_map = {0: "Normal(正常)", 1: "Diabetes(糖尿病)", 2: "Glaucoma(青光眼)",
                     3: "Cataract(白内障)", 4: "AMD", 5: "Hypertension(高血压)",
                     6: "Myopia(近视)", 7: "Others(其他眼疾)"}

        with torch.no_grad():
            logits = model_classify(classify_img_l_batch, classify_img_r_batch)
            pred = torch.stack([torch.sigmoid(logits[label].squeeze(1)) for label in logits.keys()], dim=1)

        # 显存回收
        del model_classify
        gc.collect()
        torch.cuda.empty_cache()

        # 获取各眼疾概率
        Probability = {}  # 各疾病概率
        for i in range(classify_img_l_batch.shape[0]):
            patient = {}
            j = 0
            for label in logits.keys():
                patient[label] = pred[i, j].item()
                j += 1
                Probability[str(i)] = patient

        # 获取眼疾预测结果
        pred = (pred > 0.5).float()
        predict_results = []        # 预测结果
        for i in range(pred.shape[0]):
            predict_result = ([label_map[j] for j in range(pred.shape[1]) if pred[i][j] == 1])
            predict_results.append(predict_result)

        # segment
        segment_img_l_list = img_l_list
        segment_img_r_list = img_r_list

        # 预处理 - left
        segment_img_l_list = [Grayscale()(img) for img in segment_img_l_list]
        segment_img_l_list = [ToTensor()(img) for img in segment_img_l_list]
        segment_img_l_list = [Resize((seg_img_size, seg_img_size))(img) for img in segment_img_l_list]
        norm_segment_img_l_list = segment_normalization(segment_img_l_list)

        # 预处理 - right
        segment_img_r_list = [Grayscale()(img) for img in segment_img_r_list]
        segment_img_r_list = [ToTensor()(img) for img in segment_img_r_list]
        segment_img_r_list = [Resize((seg_img_size, seg_img_size))(img) for img in segment_img_r_list]
        norm_segment_img_r_list = segment_normalization(segment_img_r_list)

        # 分割

        # 批量堆叠
        # left
        segment_img_l_batch = torch.stack(norm_segment_img_l_list, dim=0).to(device)
        # right
        segment_img_r_batch = torch.stack(norm_segment_img_r_list, dim=0).to(device)

        # 获取分割结果
        with torch.no_grad():
            output_l = model_segment(segment_img_l_batch)
            output_r = model_segment(segment_img_r_batch)
            pred_l = torch.sigmoid(output_l)
            pred_r = torch.sigmoid(output_r)

        # 生成置信度图
        confidence_map_l_list = [pred_l.cpu().detach().numpy() for pred_l in pred_l]
        confidence_map_l_list = [np.transpose(confidence_map_l, (1, 2, 0)) for confidence_map_l in confidence_map_l_list]
        confidence_map_r_list = [pred_r.cpu().detach().numpy() for pred_r in pred_r]
        confidence_map_r_list = [np.transpose(confidence_map_r, (1, 2, 0)) for confidence_map_r in confidence_map_r_list]

        confidence_map_l_file_name_list = []
        for confidence_map_l in confidence_map_l_list:
            confidence_map_l_file_name = generate_confidence_map(confidence_map_l)
            confidence_map_l_file_name_list.append(confidence_map_l_file_name)

        confidence_map_r_file_name_list = []
        for confidence_map_r in confidence_map_r_list:
            confidence_map_r_file_name = generate_confidence_map(confidence_map_r)
            confidence_map_r_file_name_list.append(confidence_map_r_file_name)

        confidence_map_l_list = []
        for confidence_map_l_file_name in confidence_map_l_file_name_list:
            confidence_map_l = Image.open(f'{temp_dir}/{confidence_map_l_file_name}')
            confidence_map_l_list.append(confidence_map_l)

        confidence_map_r_list = []
        for confidence_map_r_file_name in confidence_map_r_file_name_list:
            confidence_map_r = Image.open(f'{temp_dir}/{confidence_map_r_file_name}')
            confidence_map_r_list.append(confidence_map_r)

        # 合并置信度图
        merged_confidence_map_list = []
        for confidence_map_l, confidence_map_r in zip(confidence_map_l_list, confidence_map_r_list):
            merged_confidence_map = merge_image(confidence_map_l, confidence_map_r)
            merged_confidence_map_list.append(merged_confidence_map)
        # 生成合并后置信度图 URL
        merged_confidence_map_url_list = []
        for merged_confidence_map in merged_confidence_map_list:
            merged_confidence_map_url = save_and_generate_url(merged_confidence_map)
            merged_confidence_map_url_list.append(merged_confidence_map_url)

        # 生成二值化分割图像
        # segment img_l
        segmented_image_l_list = [(pred_l >= 0.5).float() for pred_l in pred_l]
        segmented_image_l_list = [segmented_image_l.cpu().numpy() for segmented_image_l in segmented_image_l_list]
        segmented_image_l_list = [np.transpose(segmented_image_l, (1, 2, 0)).astype(np.uint8) for segmented_image_l in segmented_image_l_list]
        segmented_image_l_list = [Image.fromarray(segmented_image_l.squeeze() * 255) for segmented_image_l in segmented_image_l_list]

        # segment img_r
        segmented_image_r_list = [(pred_r >= 0.5).float() for pred_r in pred_r]
        segmented_image_r_list = [segmented_image_r.cpu().numpy() for segmented_image_r in segmented_image_r_list]
        segmented_image_r_list = [np.transpose(segmented_image_r, (1, 2, 0)).astype(np.uint8) for segmented_image_r in segmented_image_r_list]
        segmented_image_r_list = [Image.fromarray(segmented_image_r.squeeze() * 255) for segmented_image_r in segmented_image_r_list]

        # 合并 segment image
        merged_segmented_image_list = []
        for segmented_image_l, segmented_image_r in zip(segmented_image_l_list, segmented_image_r_list):
            merged_segmented_image = merge_image(segmented_image_l, segmented_image_r)
            merged_segmented_image_list.append(merged_segmented_image)

        # 生成合并后分割图 URL
        merged_segmented_image_url_list = []
        for merged_segmented_image in merged_segmented_image_list:
            merged_segmented_image_url = save_and_generate_url(merged_segmented_image)
            merged_segmented_image_url_list.append(merged_segmented_image_url)

        # 后处理并返回结果
        result = {
            'Probability': Probability,
            'predict_results': predict_results,
            'merged_img_url_list': merged_img_url_list,
            'merged_grad_cam_img_url_list': merged_grad_cam_img_url_list,
            'merged_confidence_map_url_list': merged_confidence_map_url_list,
            'merged_segmented_image_url_list': merged_segmented_image_url_list
        }

        app.logger.debug(f"result: {result}")
        return jsonify(result), 200

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Image download failed: {str(e)}")
        return jsonify({'error': f'Image download failed: {str(e)}'}), 400
    except Exception as e:
        app.logger.exception("Prediction error")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# 启动服务
if __name__ == '__main__':
    load_model()  # 加载分类和分割模型
    app.run(host='0.0.0.0', port=5003)  # 生产环境关闭debug
