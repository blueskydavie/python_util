import os
import numpy as np
from PIL import Image, ImageOps
from typing import List
import torch
import towhee
from utils.Utils import create_milvus_collection

from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from timm.models.factory import create_model
from OpManager import op_manager
from datetime import datetime
from multiprocessing.pool import ThreadPool
from utils.Utils import logger

# 设置模型下载路径
os.environ['TORCH_HOME'] = './model_cache'  # 相对路径
# 或者使用绝对路径
# os.environ['TORCH_HOME'] = '/path/to/your/model_cache'  # Linux
# os.environ['TORCH_HOME'] = 'C:\\path\\to\\your\\model_cache'  # Windows

class ModelUtil():
    def __init__(
        self,
        model_name: str,
        device: str=None,
        num_classes: int=1000,
        pth_path: str=None,
        threads_count: int=150
    ):
        print("[{}][{}] Loading checkpoint {} of {} to {}, num_classs: {}".format(datetime.now(), type(self).__name__, pth_path, model_name, device, num_classes))
        self.model_name = model_name
        self.device = torch.device(device)
        
        # 添加下载路径日志
        cache_dir = os.environ.get('TORCH_HOME', '~/.cache/torch')
        print(f"Model will be downloaded to: {os.path.expanduser(cache_dir)}")
        
        self.model = create_model(model_name, pretrained=True, num_classes=num_classes)
        if pth_path is not None:
            self.model.load_state_dict(torch.load(pth_path))
        self.model.eval()
        self.model.to(self.device)
        self.config = resolve_data_config({}, model=self.model)
        self.tfms = create_transform(**self.config)
        self.threads_count = threads_count
        self.pool = ThreadPool(processes=self.threads_count)

    def img2vec(self, imgs_: List[Image.Image]) -> np.ndarray:
        imgs = [self.tfms(img.convert("RGB")) for img in imgs_]
        inputs = torch.stack(imgs)
        inputs = inputs.to(self.device)
        features = self.model.forward_features(inputs)
        global_pool = torch.nn.AdaptiveAvgPool2d(1)
        features = global_pool(features)
        features = features.flatten(1)
        vecs = features.cpu().squeeze(0).detach().numpy()
        return vecs


    def gray_preprocess(self, imgs_: List[Image.Image]):
        def func(img):
            tmp = img.convert("L")
            tmp = ImageOps.colorize(tmp, (0, 0, 0), (255, 255, 255))
            return tmp
        imgs = self.pool.map(func, imgs_)
        return imgs



    def to_match(self):
        self.device = torch.device("cpu")
        self.model.to(self.device)


    def to_insert(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)
        else:
            logger.warn("CUDA is not available.")



    def path2vec(self, path: str):
        tensor = self.tfms(Image.open(path))
        vec = self.img2vec([tensor])
        return vec


    def img2cluster(self, imgs):
        inputs = torch.stack(imgs)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out = self.model(inputs)
            _, pre = torch.max(out.data, 1)
            return pre.cpu().detach().numpy()


def custom_match(
        path: str, 
        clt_name: str,
        model: ModelUtil,
        limit: int=500
        ):
    clt = create_milvus_collection(collection_name=clt_name, overwrite=False)
    res = (
            towhee
            .glob["path"](path)
            .runas_op["path", "vec"](func=model.path2vec)
            .milvus_search["vec", "result"](collection=clt, limit=limit, output_fields=["id", "path"])
            )
    re = []
    avoid_repeat = []
    for item in res.to_list()[0].result:
        if item.path in avoid_repeat:
            continue
        avoid_repeat.append(item.path)
        re.append({
            "path": item.path,
            "compressed": item.path,
            "score": item.score
            })
    return re, res.to_list()[0].result




if __name__ == "__main__":
    for k, v in op_manager.ops.items():
        m1 = v()
    pass

# resnext101_class_5 = ModelUtil(
#         model_name="resnext101_32x8d",
#         device="cpu",
#         num_classes=5,
#         pth_path="../custom-model/model/fabric-class-5/final_epoch/model.pth"
#         )



