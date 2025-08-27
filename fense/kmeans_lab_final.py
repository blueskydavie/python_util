import random
import time

# 使用LAB空间进行聚类
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image  # 用于保存 TIFF 格式
import os

def split_color(image_file, file_name, dir_path, color_num=13):
    dest_path = dir_path+file_name
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    # 加载图像
    image = cv2.imread(image_file)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # 转换为 LAB 颜色空间
    print('image_lab shape:', image_lab.shape)

    # 将图像展平为二维数组 (n_pixels, 3)
    pixels = image_lab.reshape(-1, 3).astype(np.float32)
    print('pixels shape:', pixels.shape)

    # 归一化 LAB 颜色，避免 L, A, B 取值范围不同的影响
    pixels[:, 0] /= 255.0  # L 通道归一化（原范围 0-255）
    pixels[:, 1:] = (pixels[:, 1:] - 128) / 128  # A, B 通道归一化（原范围 -128~127）

    # 使用 KMeans 进行聚类
    n_clusters = color_num  # 设置聚类数量
    # kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    #, init='k-means++' 是默认操作，可以不写。
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(pixels)

    # 获取聚类结果
    labels = kmeans.labels_
    print('labels shape:', labels.shape)
    print('labels:', labels)
    cluster_centers = kmeans.cluster_centers_

    # 反归一化 LAB 颜色
    # cluster_centers[:, 0] *= 255  # 恢复 L 通道
    # cluster_centers[:, 1:] = cluster_centers[:, 1:] * 128 + 128  # 恢复 A, B 通道
    # cluster_centers = np.clip(cluster_centers, 0, 255).astype(np.uint8)  # 限制数值范围

    cluster_centers[:, 0] = np.clip(np.round(cluster_centers[:, 0] * 255), 0, 255)
    cluster_centers[:, 1:] = np.clip(np.round(cluster_centers[:, 1:] * 128 + 128), 0, 255)

    # 生成分色图像（LAB -> BGR -> RGB）
    segmented_lab = cluster_centers[labels].reshape(image_lab.shape).astype(np.uint8)
    segmented_bgr = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)
    segmented_rgb = cv2.cvtColor(segmented_bgr, cv2.COLOR_BGR2RGB)

    # 保存分色后的图像
    cv2.imwrite('segmented_image.jpg', cv2.cvtColor(segmented_rgb, cv2.COLOR_RGB2BGR))

    # 生成透明 Mask 并保存为 TIFF
    for i, color in enumerate(cluster_centers):
        mask = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)  # RGBA 格式
        mask_rgb = np.zeros_like(segmented_rgb)  # 该类别的颜色
        alpha_channel = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # 透明度

        color_bgr = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_LAB2BGR)[0][0]
        color_rgb = color_bgr[::-1]  # 转换为 RGB
        mask_rgb[labels.reshape(image.shape[:2]) == i] = color_rgb

        # 设定颜色区域
        # mask_rgb[labels.reshape(image.shape[:2]) == i] = color  # 颜色赋值
        alpha_channel[labels.reshape(image.shape[:2]) == i] = 255  # 目标区域透明度设为 255

        # 合并 RGB 和 Alpha 通道
        mask[..., :3] = mask_rgb
        mask[..., 3] = alpha_channel  # 设置 Alpha 透明度

        # 转换为 PIL 图像并保存为 TIFF
        mask_pil = Image.fromarray(mask, mode='RGBA')
        mask_filename = f'{dest_path}/lab_mask_cluster_tif_{i}.tif'
        mask_pil.save(mask_filename, format='TIFF')
        mask_filename = f'{dest_path}/lab_mask_cluster_rgb_{i}.png'
        mask_pil.save(mask_filename, format='PNG')
        print(f'Saved transparent TIFF mask: {mask_filename} for color: {color.tolist()}')

# 显示分色结果
# plt.imshow(segmented_rgb)
# plt.title("Clustered Image (LAB)")
# plt.axis("off")
# plt.show()

if __name__ == '__main__':
    # 目标文件夹路径
    folder_path = "yuyue"

    file_color_num = {
        "2": 2,
        "3": 7,
        "4": 5,
        "5": 12,
    }

    # 支持的图片格式
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}
    dest_dir = 'output/'
    # 遍历文件夹及其子文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                file_name = file.split('.')[0]
                color_num = file_color_num.get(file_name, 13)
                print(file.split('.')[0])  # 只打印文件名
                begin_time = time.time()
                split_color(os.path.join(root, file), file_name, dest_dir, color_num)
                print(f'{file_name} consume Time:', time.time() - begin_time)

    # image_file = 'test5.jpg'
    # file_name = 'test5'
    # split_color(image_file, file_name)