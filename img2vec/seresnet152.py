from OpManager import op_manager
from PIL import Image
from typing import List
import numpy as np
from img2vec.img2vec import ModelUtil

@op_manager.register(name="seresnet152")
class seresnet152(ModelUtil):
    def __init__(self):
        ModelUtil.__init__(
                self,
                model_name="legacy_seresnet152",
                device="cpu",
                num_classes=1000,
                pth_path=None
            )

    def get_dim(self):
        return 2048


# @op_manager.register()
class gray_seresnet152(seresnet152):
    def img2vec(self, imgs_: List[Image.Image]) -> np.ndarray:
        imgs = self.gray_preprocess(imgs_)
        return ModelUtil.img2vec(self, imgs)