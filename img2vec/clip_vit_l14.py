import torch
import clip 
import numpy as np
from PIL import Image
from typing import List
from OpManager import op_manager
from img2vec.img2vec import ModelUtil
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model
import os

@op_manager.register(name="clip_vit_l14")
class clip_vit_l14(ModelUtil):

    def __init__(self, device=None, num_classes=1000, pretrained=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # 添加下载路径日志
        cache_dir = os.environ.get('TORCH_HOME', '~/.cache/torch')
        print(f"Model will be downloaded to: {os.path.expanduser(cache_dir)}")

        # 用 timm 加载 clip-vit-l14
        self.model = create_model(
            'vit_large_patch14_clip_336',  # timm 的模型名字
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.model.eval()
        self.model.to(self.device)

        # 数据预处理配置
        self.config = resolve_data_config({}, model=self.model)
        self.tfms = create_transform(**self.config)
    
    
    def img2vec(self, imgs_: List[Image.Image]) -> np.ndarray:
        imgs = [self.tfms(img.convert("RGB")) for img in imgs_]
        inputs = torch.stack(imgs).to(self.device)
        
        features = self.model.forward_features(inputs)  # (B, seq_len, C)
        
        if len(features.shape) == 3:  # ViT 类型
            # features = features.mean(dim=1)  # 对 sequence 维度取平均
            # 取第一个 token，即 class token
            features = features[:, 0, :]
        else:  # CNN 特征
            global_pool = torch.nn.AdaptiveAvgPool2d(1)
            features = global_pool(features)
            features = features.flatten(1)
        
        # vecs = features.cpu().detach().numpy()
        vecs = features.cpu().squeeze(0).detach().numpy()
        return vecs



    # def __init__(self):
    #     # 初始化CLIP模型
    #     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)
    #     self.model.eval()
    #     self.config = resolve_data_config({}, model=self.model)
    #     self.tfms = create_transform(**self.config)
        
    # def img2vec(self, imgs_: List[Image.Image]) -> np.ndarray:
    #     """将图像转换为向量"""
    #     # 预处理图像
    #     processed_imgs = []
    #     for img in imgs_:
    #         # 确保图像是RGB格式
    #         if img.mode != 'RGB':
    #             img = img.convert('RGB')
    #         processed_img = self.preprocess(img).unsqueeze(0)
    #         processed_imgs.append(processed_img)
        
    #     # 批量处理
    #     inputs = torch.cat(processed_imgs, dim=0).to(self.device)
        
    #     with torch.no_grad():
    #         # 获取图像特征
    #         image_features = self.model.encode_image(inputs)
    #         # 归一化特征
    #         image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
    #     # 转换为numpy数组
    #     vecs = image_features.cpu().detach().numpy()
    #     return vecs
    
    def get_dim(self):
        """返回特征维度"""
        return 1024  # CLIP ViT-L/14的特征维度是768
    
    def to_match(self):
        """切换到匹配模式（CPU）"""
        self.device = torch.device("cpu")
        self.model.to(self.device)
    
    def to_insert(self):
        """切换到插入模式（GPU如果可用）"""
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
    
    def path2vec(self, path: str):
        """从路径加载图像并转换为向量"""
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        vec = self.img2vec([img])
        return vec 