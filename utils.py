import os
import random
import re

import numpy as np
import torch
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, spearmanr
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, LinearLR
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.transforms import functional as F
from sklearn.metrics import precision_recall_fscore_support

from ImageDataset import ImageDataset


def tensor2image(tensor):
    tensor = tensor.cpu().numpy()
    return Image.fromarray(tensor.astype(np.uint8))


def image2tensor(image):
    transform = T.ToTensor()
    return transform(image).squeeze(0)


def load_image_for_score(filenames):  # [5]
    images = []
    image_root_dir = "./Data_richhf18k/richhf_18k/image"

    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
    resize = T.Resize((224, 224))
    crop_box = random_crop()

    for filename in filenames:
        folder_name = filename.split('/')[0]  # must be "train"
        image_path = os.path.join(image_root_dir, folder_name, filename.split('/')[-1])

        image = Image.open(image_path).convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512))
        image = image.crop(crop_box)
        image = image_augmentations(image)
        image = to_tensor(image)
        image = normalize(image)
        image = resize(image)
        images.append(image)  # [5, C=3, H=224, W=224]

    return images  # [5, C, H, W]


def load_image_and_prompt(filenames, artifact_maps, misalignment_maps):
    """
    从指定路径加载images prompts

    参数:
        filenames (list of str): 文件名列表，文件名格式如 'test/bf07713d-b61a-4323-9515-7e9c4a70253b.png'

    返回:
        images (list of PIL.Image): 图像列表
        prompts (list of str): prompt 列表
    """
    images = []
    prompts = []
    artifact_map_tmp = []
    misalignment_map_tmp = []
    image_root_dir = "./Data_richhf18k/richhf_18k/image"
    prompt_root_dir = "./Data_richhf18k/richhf_18k/prompt"

    to_tensor = T.ToTensor()
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
    resize = T.Resize((224, 224))

    crop_box = random_crop()  # 获取这个batch的裁剪尺寸

    for filename in filenames:
        folder_name = filename.split('/')[0]

        # 构造图像文件的路径
        image_path = os.path.join(image_root_dir, folder_name, filename.split('/')[-1])

        # 加载图像
        image = Image.open(image_path).convert("RGB")  # 转为 RGB 格式，适用于彩色图像
        if image.size != (512, 512):
            image = image.resize((512, 512))

        if folder_name == "train":
            image = image.crop(crop_box)
            image = image_augmentations(image)
        image = to_tensor(image)
        image = normalize(image)
        image = resize(image)
        images.append(image)

        artifact_map = artifact_maps[filenames.index(filename)]
        misalignment_map = misalignment_maps[filenames.index(filename)]
        if folder_name == "train" and crop_box is not None:
            # 将 artifact_map 和 misalignment_map 转换为 PIL 图像
            artifact_map_img = tensor2image(artifact_map)
            misalignment_map_img = tensor2image(misalignment_map)
            # 对 artifact_map 和 misalignment_map 进行裁剪操作
            artifact_map_img = artifact_map_img.crop(crop_box)
            misalignment_map_img = misalignment_map_img.crop(crop_box)
            # 将裁剪后的图像还原为 Tensor
            artifact_map = image2tensor(artifact_map_img)
            misalignment_map = image2tensor(misalignment_map_img)
            # resize为与image同样的size
        artifact_map = resize(artifact_map.unsqueeze(0))
        misalignment_map = resize(misalignment_map.unsqueeze(0))
        artifact_map_tmp.append(artifact_map.unsqueeze(0))
        misalignment_map_tmp.append(misalignment_map.unsqueeze(0))

        # 构造对应 prompt 的文件路径
        prompt_filename = filename.split('/')[-1] + '.txt'  # 假设 prompt 文件与文件夹名相同
        prompt_path = os.path.join(prompt_root_dir, folder_name, prompt_filename)

        # 读取 prompt 内容
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        prompts.append(prompt)

    return images, prompts, artifact_map_tmp, misalignment_map_tmp


def random_crop(min_scale=0.8, max_scale=1.0):
    """
    随机裁剪图像，裁剪区域的宽度和高度在 min_scale 到 max_scale 之间。
    """
    if random.random() < 0.5:  # 50% 概率裁剪
        width, height = 512, 512
        rate = random.uniform(min_scale, max_scale)
        target_width = rate * width
        target_height = rate * height

        left = random.uniform(0, width - target_width)
        top = random.uniform(0, height - target_height)
        right = left + target_width
        bottom = top + target_height

        crop_box = (left, top, right, bottom)
        # image = image.crop(crop_box)
        return crop_box
    else:
        return None


# 定义随机JPEG噪声
def random_jpeg_noise(image, min_quality=70, max_quality=100):
    """
    随机对图像应用JPEG噪声，使用不同的JPEG质量（从70到100）。
    """
    if random.random() < 0.1:  # 10% 概率应用 JPEG 噪声
        quality = random.randint(min_quality, max_quality)
        image = apply_jpeg_noise(image, quality)
    return image


def apply_jpeg_noise(image, quality):
    """
    将JPEG噪声应用到图像：将图像保存为JPEG格式并以指定质量重新加载。
    """
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    image = Image.open(buffer)
    return image


# 定义随机灰度转换
def random_grayscale(image, p=0.1):
    """
    以 p 的概率将图像转换为灰度。
    """
    if random.random() < p:  # 10% 概率转换为灰度
        image = image.convert("L")  # 转为灰度图
        image = image.convert("RGB")  # 再转回 RGB 图
    return image


# 定义主图像增强函数
def image_augmentations(image):
    """
    应用一系列随机增强到图像：裁剪，亮度调整，对比度调整，色调和饱和度调整，
    JPEG 噪声和灰度转换。
    """

    # 随机亮度、对比度、色调、饱和度调整
    color_jitter = T.ColorJitter(brightness=0.05, contrast=0.2, saturation=0.2, hue=0.025)
    image = color_jitter(image)

    # 随机 JPEG 噪声
    image = random_jpeg_noise(image, min_quality=70, max_quality=100)

    # 随机转换为灰度
    image = random_grayscale(image, p=0.1)

    # 随机水平翻转
    if random.random() < 0.5:  # 50% 概率进行水平翻转
        image = F.hflip(image)

    return image


# def display_overlay_images(images, artifact_maps):
#     """
#     显示原图和叠加后的图像。
#
#     参数:
#         images (list of PIL.Image): 原始图像列表。
#         artifact_maps (list of Tensor): artifact_map 列表。
#     """
#     for idx, image in enumerate(images):
#         artifact_map = artifact_maps[idx].squeeze(0)  # 去掉 batch 维度
#
#         # 将原图和 artifact_map 叠加
#         overlay_image = overlay_artifact_map(image, artifact_map, alpha=0.6)
#
#         # 显示叠加后的结果
#         plt.figure(figsize=(10, 5))
#
#         # 原始图像
#         plt.subplot(1, 2, 1)
#         # plt.title("Original Image")
#         plt.imshow(image.permute(1, 2, 0))
#         plt.axis("off")
#
#         # 叠加后的图像
#         plt.subplot(1, 2, 2)
#         # plt.title("Overlay Image")
#         plt.imshow(overlay_image)
#         plt.axis("off")
#
#         # 显示图像
#         plt.show()


def calculate_pixelwise_mse_batch(predictions, targets):
    """
    计算批量输入的像素级 MSE
    Args:
        predictions: 模型输出的热图 (torch.Tensor, shape: [B, H, W] 或 [B, C, H, W])
        targets: 真实热图 (torch.Tensor, shape: [B, H, W] 或 [B, C, H, W])
    Returns:
        mse_mean: 整个 batch 的平均 MSE (float)
    """
    # 每个样本的 MSE
    mse_per_sample = torch.nn.functional.mse_loss(predictions, targets, reduction='none')  # 不做平均
    mse_per_sample = mse_per_sample.view(mse_per_sample.size(0), -1).mean(dim=1)  # 按样本求平均

    # 整个 batch 的平均 MSE
    mse_mean = mse_per_sample.mean().item()

    return mse_mean


def clean_sentences(prompts):
    # 定义所有分隔符的字符，包括标点符号和空格
    delimiters = ',.?!":; '

    # 构建正则表达式，用于匹配所有分隔符
    pattern = '|'.join(map(re.escape, delimiters))

    # 存储处理后的句子
    cleaned_sentences = []

    for prompt in prompts:
        # 使用正则表达式拆分输入字符串，并去除空字符串
        tokens = re.split(pattern, prompt)
        tokens = [t for t in tokens if t]  # 去除空字符串

        # 将有效字符重新组合成一个句子
        cleaned_sentence = ' '.join(tokens)
        cleaned_sentences.append(cleaned_sentence)

    return cleaned_sentences


def rank_loss(scores):
    n = len(scores)
    loss = 0.0
    pair_count = 0
    for j in range(n):
        for l in range(j + 1, n):
            xi = scores[j]
            xj = scores[l]
            score_diff = xi - xj
            prob = torch.sigmoid(score_diff)
            loss -= torch.log(prob)
            pair_count += 1
    return loss / pair_count


def lr_lambda(current_step: int, warmup_iterations: int, base_lr: float):
    if current_step <= warmup_iterations:
        return max(current_step / warmup_iterations * base_lr, 1e-3)  # 线性升高学习率
    return base_lr / current_step ** 0.5  # 递减学习率


def get_optimizer_and_scheduler(model, base_learning_rate=0.015, warmup_iterations=4):
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    #                         lr=base_learning_rate,
    #                         weight_decay=1e-4)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=base_learning_rate,
                          momentum=0.9,
                          weight_decay=5e-4)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_lambda(step, warmup_iterations, base_learning_rate))
    return optimizer, scheduler


def get_optimizer_and_scheduler_cosine(model, max_lr=1e-3, warmup_epochs=4, total_epochs=12):
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=max_lr,
                            weight_decay=1e-3)

    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    return optimizer, scheduler


if __name__ == '__main__':
    bs = 256
    train_dir = "./Data_richhf18k/torch/train"
    dev_dir = "./Data_richhf18k/torch/dev"
    test_dir = "./Data_richhf18k/torch/test"
    train_dataset = ImageDataset(train_dir)
    dev_dataset = ImageDataset(dev_dir)
    test_dataset = ImageDataset(test_dir)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    for batch in train_loader:
        # artifact_maps, misalignment_maps, scores, token_labels = batch
        filenames = batch['filename']
        artifact_maps = batch['artifact_map']
        misalignment_maps = batch['misalignment_map']
        scores = batch['scores']
        token_labels = batch['token_label']

        images, prompts, artifact_maps, misalignment_maps = load_image_and_prompt(filenames, artifact_maps,
                                                                                  misalignment_maps)
        for i in range(len(token_labels)):
            print(prompts[i], "\n", token_labels[i], "\n")

        break
